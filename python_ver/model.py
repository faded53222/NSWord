from imports import *

def glorot_uniform_init_(weights):
	nn.init.xavier_uniform_(weights,gain=1)
def zero_init_(weights):
	with torch.no_grad():
		weights.fill_(0.0)
def permute_final_dims(tensor,inds):
	zero_index=-1*len(inds)
	first_inds=list(range(len(tensor.shape[:zero_index])))
	return tensor.permute(first_inds+[zero_index+i for i in inds])
def flatten_final_dims(t,no_dims):
	return t.reshape(t.shape[:-no_dims]+(-1,))
def relu_init_(weights,scale=2.0):
	shape=weights.shape
	_,f=shape
	scale=scale/max(1,f)
	a=-2
	b=2
	std=math.sqrt(scale)/truncnorm.std(a=a,b=b,loc=0,scale=1)
	size=1
	for n in shape:
		size=size*n
	samples=truncnorm.rvs(a=a,b=b,loc=0,scale=std,size=size)
	samples=np.reshape(samples,shape)
	with torch.no_grad():
		weights.copy_(torch.tensor(samples,device=weights.device))

class Linear(nn.Linear):
	def __init__(self,in_dim,out_dim,bias=True,init="zero"):
		super(Linear,self).__init__(in_dim,out_dim,bias=bias)
		if bias:
			with torch.no_grad():
				self.bias.fill_(0)
		with torch.no_grad():
			if init=='zero':
				zero_init_(self.weight)
			elif init=='glorot':
				glorot_uniform_init_(self.weight)
			elif init=='relu':
				relu_init_(self.weight)
			elif init=='gating':
				zero_init_(self.weight)
				if bias:
					self.bias.fill_(1.0)
			else:
				 glorot_uniform_init_(self.weight)

class LayerNorm(nn.Module):
	def __init__(self,c_in,eps=1e-5):
		super(LayerNorm, self).__init__()
		self.c_in=(c_in,)
		self.eps=eps
		self.weight=nn.Parameter(torch.ones(c_in))
		self.bias=nn.Parameter(torch.zeros(c_in))
	def forward(self,x): 
		out=nn.functional.layer_norm(x,self.c_in,self.weight,self.bias,self.eps)
		return out

class Dropout(nn.Module):
	def __init__(self,r,batch_dim):
		super(Dropout,self).__init__()
		self.r=r
		if type(batch_dim)==int:
			batch_dim=[batch_dim]
		self.batch_dim=batch_dim
		self.dropout=nn.Dropout(r)
	def forward(self,x):
		shape=list(x.shape)
		if self.batch_dim is not None:
			for bd in self.batch_dim:
				shape[bd]=1
		mask=x.new_ones(shape)
		mask=self.dropout(mask)
		x*=mask
		return x
class DropoutRowwise(Dropout):
	__init__=partialmethod(Dropout.__init__,batch_dim=-3)
class DropoutColwise(Dropout):
	__init__=partialmethod(Dropout.__init__,batch_dim=-2)

class LinearEmbedder(nn.Module):
	def __init__(self,c_in,c_out):
		super(LinearEmbedder,self).__init__()
		self.c_in=c_in
		self.c_out=c_out
		self.linear_1=Linear(c_in,c_out,init='relu')
		self.relu=nn.ReLU()
		self.linear_2=nn.Linear(c_out,c_out)
	def forward(self,x):
		x=self.linear_1(x)
		x=self.relu(x)
		x=self.linear_2(x)
		return x

class Transition(nn.Module):
	def __init__(self,c_x,transition_n=2):
		super(Transition,self).__init__()
		self.LayerNorm_trans=LayerNorm(c_x)
		self.linear_1=nn.Linear(c_x,c_x*transition_n)
		self._elu=nn.ELU()
		#self.linear_1=Linear(c_x,c_x*transition_n,init='relu')
		#self._elu=nn.ReLU()
		self.linear_2=Linear(c_x*transition_n,c_x,init='zero')
	def forward(self,x):
		x=self.LayerNorm_trans(x)
		x=self.linear_1(x)
		x=self._elu(x)
		x=self.linear_2(x)
		return x

MAX_SEQ_LEN=50
def precompute_freqs_cis(dim,seq_len,theta=10000.0):
	freqs=1.0/(theta**(torch.arange(0,dim,2)[:(dim//2)].float()/dim))
	t=torch.arange(seq_len,device=freqs.device)
	freqs=torch.outer(t,freqs).float()
	freqs_cis=torch.polar(torch.ones_like(freqs),freqs)
	return freqs_cis

def apply_rotary_emb(q,k,freqs_cis,same=True):
	_q=q.float().reshape(*q.shape[:-1],-1,2)
	_k=k.float().reshape(*k.shape[:-1],-1,2)
	_q=torch.view_as_complex(_q)
	_k=torch.view_as_complex(_k)
	
	if same==False:
		if _k.shape[-2]%2!=0:
			q_out=torch.view_as_real(_q*freqs_cis[int((_k.shape[-2]-1)/2)].to(q.device)).flatten(-2)
		else:
			q_out=torch.view_as_real(_q*freqs_cis[_k.shape[-2]/2].to(q.device)).flatten(-2)
	else:
		q_out=torch.view_as_real(_q*freqs_cis[:_q.shape[-2]].to(q.device)).flatten(-2)
	k_out=torch.view_as_real(_k*freqs_cis[:_k.shape[-2]].to(k.device)).flatten(-2)
	return q_out.type_as(q),k_out.type_as(k)

class Attention(nn.Module):
	def __init__(self,c_q,c_k,c_v,c_hidden,no_heads,gating=True,use_rel_pos=False):
		super(Attention, self).__init__()
		self.c_q=c_q
		self.c_k=c_k
		self.c_v=c_v
		self.c_hidden=c_hidden
		self.no_heads=no_heads
		self.gating=gating
		self.use_rel_pos=use_rel_pos

		self.linear_q=Linear(c_q,c_hidden*no_heads,bias=False,init='glorot')
		self.linear_k=Linear(c_k,c_hidden*no_heads,bias=False,init='glorot')
		self.linear_v=Linear(c_v,c_hidden*no_heads,bias=False,init='glorot')
		self.linear_o=Linear(c_hidden*no_heads,c_q,init='zero')
		if self.gating:
			self.linear_g=Linear(c_q,c_hidden*no_heads,init='gating')
		self.sigmoid=nn.Sigmoid()

		self.freqs_cis=None
		if self.use_rel_pos:
			self.freqs_cis=precompute_freqs_cis(c_hidden,MAX_SEQ_LEN)

	def forward(self,q_x,kv_x,biases=None):
		if(biases is None):
			biases=[]
		q=self.linear_q(q_x)
		k=self.linear_k(kv_x)
		v=self.linear_v(kv_x)
		q=q.view(q.shape[:-1]+(self.no_heads,-1))
		k=k.view(k.shape[:-1]+(self.no_heads,-1))
		v=v.view(v.shape[:-1]+(self.no_heads,-1))

		q=q.transpose(-2,-3)#r,H,s,h
		k=k.transpose(-2,-3)
		v=v.transpose(-2,-3)
		
		if self.use_rel_pos:
			q,k=apply_rotary_emb(q,k,freqs_cis=self.freqs_cis,same=True)
		k=permute_final_dims(k,(1,0))
		a=torch.matmul(q,k)/math.sqrt(self.c_hidden)#r,H,s,h * r,H,h,s = r,H,s,s
		for b in biases:
			a+=b
		a=torch.nn.functional.softmax(a,dim=-1)
		o=torch.matmul(a,v)#r,H,s,s * r,H,s,h = r,H,s,h
		o=o.transpose(-2,-3)#r,s,H,h

		if self.gating:
			g=self.sigmoid(self.linear_g(q_x))
			g=g.view(g.shape[:-1]+(self.no_heads,-1))
			o=o*g
		o=flatten_final_dims(o,2)#r,s,H*h
		o=self.linear_o(o)#r,s,o
		return o

class NanoAttention(nn.Module):
	def __init__(self,c_in,c_hidden,no_heads,inf=1e9,use_rel_pos=False):
		super(NanoAttention,self).__init__()
		self.c_in=c_in
		self.c_hidden=c_hidden
		self.no_heads=no_heads
		self.inf=inf
		self.use_rel_pos=use_rel_pos
		self.layer_norm_x=LayerNorm(c_in)
		self.mha=Attention(c_in,c_in,c_in,c_hidden,no_heads,True,use_rel_pos)

	def forward(self,x,mask=None):
		n_seq,n_pos=x.shape[-3:-1]
		if mask is None:
			mask=x.new_ones(x.shape[:-3]+(n_seq,n_pos))
		mask_bias=(self.inf*(mask-1))[...,:,None,None,:]
		biases=[mask_bias]

		x=self.layer_norm_x(x)
		x=self.mha(x,x,biases)
		return x

class Trans_NanoAttention(nn.Module):
	def __init__(self,c_in,c_hidden,no_heads,inf=1e9,use_rel_pos=False):
		super(Trans_NanoAttention,self).__init__()
		self.c_in=c_in
		self.c_hidden=c_hidden
		self.no_heads=no_heads
		self.inf=inf
		self.use_rel_pos=use_rel_pos
		self._NanoAttention=NanoAttention(c_in,c_hidden,no_heads,inf,use_rel_pos)

	def forward(self,x,mask=None):
		x=x.transpose(-2,-3)
		if mask is not None:
			mask=mask.transpose(-1,-2)
		x=self._NanoAttention(x,mask=mask)

		x=x.transpose(-2,-3)
		if mask is not None:
			mask=mask.transpose(-1,-2)
		return x

class GlobalAttention(nn.Module):
	def __init__(self,c_in,c_hidden,no_heads,inf=1e5,eps=1e-8,use_rel_pos=False):
		super(GlobalAttention,self).__init__()
		self.c_in=c_in
		self.c_hidden=c_hidden
		self.no_heads=no_heads
		self.inf=inf
		self.eps=eps
		self.use_rel_pos=use_rel_pos
		
		self.linear_q=Linear(c_in,c_hidden*no_heads,bias=False,init='glorot')
		self.linear_k=Linear(c_in,c_hidden,bias=False,init='glorot')
		self.linear_v=Linear(c_in,c_hidden,bias=False,init='glorot')
		self.linear_g=Linear(c_in,c_hidden*no_heads,init='gating')
		self.linear_o=Linear(c_hidden*no_heads,c_in,init='zero')
		self.sigmoid=nn.Sigmoid()
		self.freqs_cis=None
		if self.use_rel_pos:
			self.freqs_cis=precompute_freqs_cis(c_hidden,MAX_SEQ_LEN)
	def forward(self,m,mask):
		q=torch.sum(m*mask.unsqueeze(-1),dim=-2)/(torch.sum(mask,dim=-1)[...,None]+self.eps)
		q=self.linear_q(q)
		k=self.linear_k(m)#r,s,h
		v=self.linear_v(m)#r,s,h
		q=q.view(q.shape[:-1]+(self.no_heads,-1))#r,H,h
		if self.use_rel_pos:
			q,k=apply_rotary_emb(q,k,freqs_cis=self.freqs_cis)
		
		bias=(self.inf*(mask-1))[...,:,None,:]
		a=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(self.c_hidden)#r,H,h * r,h,s = r,H,s
		a+=bias
		a=torch.nn.functional.softmax(a,dim=-1)
		
		o=torch.matmul(a,v)#r,H,s * r,s,h = r,H,h
		g=self.sigmoid(self.linear_g(m))
		g=g.view(g.shape[:-1]+(self.no_heads,-1))
		o=o.unsqueeze(-3)*g#r,1,H,h * r,s,H,h = r,s,H,h
		o=o.reshape(o.shape[:-2]+(-1,))
		
		m=self.linear_o(o)#r,s,H*h->r,s,c_in
		return m

class GlobalNanoAttention(nn.Module):
	def __init__(self,c_in,c_hidden,no_heads,inf=1e9,eps=1e-8,use_rel_pos=False):
		super(GlobalNanoAttention,self).__init__()
		self.c_in=c_in
		self.c_hidden=c_hidden
		self.no_heads=no_heads
		self.inf=inf
		self.use_rel_pos=use_rel_pos
		self.layer_norm_x=LayerNorm(c_in)
		self.gmha=GlobalAttention(c_in,c_hidden,no_heads,inf,eps,use_rel_pos)

	def forward(self,x,mask=None):
		n_seq,n_pos=x.shape[-3:-1]
		if mask is None:
			mask=x.new_ones(x.shape[:-3]+(n_seq,n_pos))
		x=self.layer_norm_x(x)
		x=self.gmha(x,mask)
		return x

class Trans_GlobalNanoAttention(nn.Module):
	def __init__(self,c_in,c_hidden,no_heads,inf=1e9,eps=1e-8,use_rel_pos=False):
		super(Trans_GlobalNanoAttention,self).__init__()
		self.c_in=c_in
		self.c_hidden=c_hidden
		self.no_heads=no_heads
		self.inf=inf
		self.use_rel_pos=use_rel_pos
		self._GlobalNanoAttention=GlobalNanoAttention(c_in,c_hidden,no_heads,inf,eps,use_rel_pos)

	def forward(self,x,mask=None):
		x=x.transpose(-2,-3)
		if mask is not None:
			mask=mask.transpose(-1,-2)
		x=self._GlobalNanoAttention(x,mask=mask)
		x=x.transpose(-2,-3)
		if mask is not None:
			mask=mask.transpose(-1,-2)
		return x

class LineAttention(nn.Module):
	def __init__(self,c_in,c_hidden,no_heads,inf=1e5,eps=1e-8,use_rel_pos=False):
		super(LineAttention,self).__init__()
		self.c_in=c_in
		self.c_hidden=c_hidden
		self.no_heads=no_heads
		self.inf=inf
		self.eps=eps
		self.use_rel_pos=use_rel_pos
		
		self.linear_q0=Linear(c_in,c_hidden*no_heads,bias=False,init='glorot')
		self.linear_k0=Linear(c_in,c_hidden,bias=False,init='glorot')
		self.linear_v0=Linear(c_in,c_hidden,bias=False,init='glorot')
		self.linear_q1=Linear(c_hidden,c_hidden,bias=False,init='glorot')
		self.linear_k1=Linear(c_hidden,c_hidden,bias=False,init='glorot')
		self.linear_v1=Linear(c_hidden,c_hidden,bias=False,init='glorot')
		self.linear_g=Linear(c_in,c_hidden*no_heads,init='gating')
		self.linear_o=Linear(c_hidden*no_heads,c_in,init='zero')
		self.sigmoid=nn.Sigmoid()
		self.freqs_cis=precompute_freqs_cis(c_hidden,MAX_SEQ_LEN)
	def forward(self,m,mask):
		l_sum=torch.sum(m*mask.unsqueeze(-1),dim=-2)/(torch.sum(mask,dim=-1)[...,None]+self.eps)
		q0=self.linear_q0(l_sum)
		k0=self.linear_k0(m)#r,s,h
		v0=self.linear_v0(m)#r,s,h
		q0=q0.view(q0.shape[:-1]+(self.no_heads,-1))#r,H,h
		if self.use_rel_pos:
			q0,k0=apply_rotary_emb(q0,k0,freqs_cis=self.freqs_cis,same=False)#r,H,h;r,s,h
		bias=(self.inf*(mask-1))[...,:,None,:]
		a0=torch.matmul(q0,k0.transpose(-1,-2))/math.sqrt(self.c_hidden)#r,H,h * r,h,s = r,H,s
		a0+=bias
		a0=torch.nn.functional.softmax(a0,dim=-1)
		r0=torch.matmul(a0,v0)#r,H,s * r,s,h = r,H,h
		
		q1=self.linear_q1(r0)
		k1=self.linear_q1(r0)
		v1=self.linear_q1(r0)
		q1=q1.transpose(-2,-3)
		k1=k1.transpose(-2,-3)
		v1=v1.transpose(-2,-3)
		if not self.use_rel_pos:
			q1,k1=apply_rotary_emb(q1,k1,freqs_cis=self.freqs_cis,same=True)#H,r,h;H,r,h
		a1=torch.matmul(q1,k1.transpose(-1,-2))/math.sqrt(self.c_hidden)#H,r,h * H,h,r = H,r,r
		a1=torch.nn.functional.softmax(a1,dim=-1)
		r1=torch.matmul(a1,v1)#H,r,r * H,r,h = H,r,h

		g=self.sigmoid(self.linear_g(m))
		g=g.view(g.shape[:-1]+(self.no_heads,-1))
		g=g.transpose(-2,-3)
		g=g.transpose(-3,-4)

		r=r1.unsqueeze(-2)*g#H,r,1,h*H,r,s,h=H,r,s,h
		r=r.transpose(-3,-4)
		r=r.transpose(-2,-3)
		r=r.reshape(r.shape[:-2]+(-1,))
		m=self.linear_o(r)#r,s,H*h->r,s,c_in
		return m

class LineNanoAttention(nn.Module):
	def __init__(self,c_in,c_hidden,no_heads,inf=1e9,eps=1e-8,use_rel_pos=False):
		super(LineNanoAttention,self).__init__()
		self.c_in=c_in
		self.c_hidden=c_hidden
		self.no_heads=no_heads
		self.inf=inf
		self.use_rel_pos=use_rel_pos
		self.layer_norm_x=LayerNorm(c_in)
		self.lmha=LineAttention(c_in,c_hidden,no_heads,inf,eps,use_rel_pos)

	def forward(self,x,mask=None):
		n_seq,n_pos=x.shape[-3:-1]
		if mask is None:
			mask=x.new_ones(x.shape[:-3]+(n_seq,n_pos))
		x=self.layer_norm_x(x)
		x=self.lmha(x,mask)
		return x

class Trans_LineNanoAttention(nn.Module):
	def __init__(self,c_in,c_hidden,no_heads,inf=1e9,eps=1e-8,use_rel_pos=False):
		super(Trans_LineNanoAttention,self).__init__()
		self.c_in=c_in
		self.c_hidden=c_hidden
		self.no_heads=no_heads
		self.inf=inf
		self.use_rel_pos=use_rel_pos
		self._LineNanoAttention=LineNanoAttention(c_in,c_hidden,no_heads,inf,eps,use_rel_pos)

	def forward(self,x,mask=None):
		x=x.transpose(-2,-3)
		if mask is not None:
			mask=mask.transpose(-1,-2)
		x=self._LineNanoAttention(x,mask=mask)
		x=x.transpose(-2,-3)
		if mask is not None:
			mask=mask.transpose(-1,-2)
		return x

class NanoBlock(nn.Module):
	def __init__(self,c_x,c_hidden_att,no_heads,dropout,transition_n,inf,eps):
		super(NanoBlock,self).__init__()
		self.att_col=Trans_NanoAttention(c_x,c_hidden_att,no_heads,inf,use_rel_pos=False)
		self.att_row=NanoAttention(c_x,c_hidden_att,no_heads,inf,use_rel_pos=True)
		self.col_dropout_layer=DropoutColwise(dropout)
		self.row_dropout_layer=DropoutRowwise(dropout)
		self.transition=Transition(c_x,transition_n)

	def forward(self,x,x_mask):
		x=x+self.col_dropout_layer(self.att_col(x,x_mask).clone())
		x=x+self.row_dropout_layer(self.att_row(x,x_mask).clone())
		x=x+self.transition(x)
		return x

class NanoGlobalBlock(nn.Module):
	def __init__(self,c_x,c_hidden_att,no_heads,dropout,transition_n,inf,eps):
		super(NanoGlobalBlock,self).__init__()
		self.att_col=Trans_NanoAttention(c_x,c_hidden_att,no_heads,inf,use_rel_pos=False)
		self.gatt_row=GlobalNanoAttention(c_x,c_hidden_att,no_heads,inf,eps,use_rel_pos=True)
		self.col_dropout_layer=DropoutColwise(dropout)
		self.row_dropout_layer=DropoutRowwise(dropout)
		self.transition=Transition(c_x,transition_n)

	def forward(self,x,x_mask):
		x=x+self.col_dropout_layer(self.att_col(x,x_mask).clone())
		x=x+self.row_dropout_layer(self.gatt_row(x,x_mask).clone())
		x=x+self.transition(x)
		return x

class NanoLineBlock(nn.Module):
	def __init__(self,c_x,c_hidden_att,no_heads,dropout,transition_n,inf,eps):
		super(NanoLineBlock,self).__init__()
		self.att_col=Trans_NanoAttention(c_x,c_hidden_att,no_heads,inf,use_rel_pos=False)
		self.latt_row=LineNanoAttention(c_x,c_hidden_att,no_heads,inf,eps,use_rel_pos=True)
		self.col_dropout_layer=DropoutColwise(dropout)
		self.row_dropout_layer=DropoutRowwise(dropout)
		self.transition=Transition(c_x,transition_n)

	def forward(self,x,x_mask):
		x=x+self.col_dropout_layer(self.att_col(x,x_mask).clone())
		x=x+self.row_dropout_layer(self.latt_row(x,x_mask).clone())
		x=x+self.transition(x)
		return x

class NanoStack(nn.Module):
	def __init__(self,c_x,c_hidden_att,no_heads,blocks_lis,
		dropout,transition_n,
		inf,eps,clear_cache_between_blocks=False):
		super(NanoStack,self).__init__()
		self.clear_cache_between_blocks=clear_cache_between_blocks
		self.blocks=nn.ModuleList()
		for block_type in blocks_lis:
			if block_type==0:
				block=NanoBlock(c_x,c_hidden_att,no_heads,dropout,transition_n,inf,eps)
			elif block_type==1:
				block=NanoGlobalBlock(c_x,c_hidden_att,no_heads,dropout,transition_n,inf,eps)
			elif block_type==2:
				block=NanoLineBlock(c_x,c_hidden_att,no_heads,dropout,transition_n,inf,eps)
			self.blocks.append(block)

	def _prep_blocks(self,x_mask):
		blocks=[partial(b,x_mask=x_mask)for b in self.blocks]
		if(self.clear_cache_between_blocks):
			def block_with_cache_clear(block,*args,**kwargs):
				torch.cuda.empty_cache()
				return block(*args,**kwargs)
			blocks=[partial(block_with_cache_clear,b) for b in blocks]
		return blocks

	def forward(self,x,x_mask):
		blocks=self._prep_blocks(x_mask)
		for block in blocks:
			x=block(x)
		return x

class Nano(nn.Module):
	def __init__(self,c_s,c_x,c_emb,c_hidden_att,c_o,no_heads,blocks_lis,
				dropout,transition_n,inf=1e9,eps=1e-8,clear_cache_between_blocks=False):
		super(Nano,self).__init__()
		self.x_embedder=LinearEmbedder(c_s+c_x,c_emb)
		self.stack=NanoStack(c_emb,c_hidden_att,no_heads,blocks_lis,
					 dropout,transition_n,inf,eps,clear_cache_between_blocks)
		t_hid=int((c_emb*c_o)**0.5)
		self.classifier=nn.Sequential(
			nn.Linear(c_emb,t_hid),
			nn.ReLU(),
			nn.Linear(t_hid,c_o),
			nn.Sigmoid()
		)
	def forward(self,s,x,s_mask,x_mask):
		s=s.unsqueeze(-3)
		s=s.expand(*[-1]*(s.dim()-3),x.shape[-3],-1,-1)

		x=torch.cat([s,x],dim=-1)
		x=self.x_embedder(x)

		x=self.stack(x,x_mask)
		x=torch.mean(x[...,:,int(x.shape[-2]/2)+1,:],-2)
		o=self.classifier(x).squeeze(-1)
		return o


