import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from shap.explainers._explainer import Explainer
class m6Anet_DeepEXP_reads(Explainer):
	def __init__(self,model,data):
		X_sum=sum(d['X'] for d in data)
		X_mean=X_sum/len(data)

		X_reads_mean=X_mean.mean(dim=0,keepdim=True)
		X_reads_mean=X_reads_mean.expand(X_mean.shape[0],-1)
		self.reads_mean={}
		self.reads_mean['X']=X_reads_mean
		
		self.model=model.eval()
		with torch.no_grad():
			outputs=[]
			for sample in data:
				c_sample=sample.copy()
				c_sample['X']=c_sample['X'].unsqueeze(0)
				c_sample['kmer']=c_sample['kmer'].unsqueeze(0)
				outputs.append(model(c_sample))
			outputs=torch.stack(outputs)
			self.num_outputs=1#1==outputs.shape[-1]
			self.device=outputs.device
			self.expected_value=torch.mean(outputs,dim=0).cpu().numpy()

	def add_handles(self,model,forward_handle,backward_handle):
		handles_list=[]
		model_children=list(model.children())
		if model_children:
			for child in model_children:
				handles_list.extend(self.add_handles(child,forward_handle,backward_handle))
		else:
			handles_list.append(model.register_forward_hook(forward_handle))
			handles_list.append(model.register_full_backward_hook (backward_handle))
		return handles_list

	def remove_attributes(self,model):
		for child in model.children():
			if 'nn.modules.container' in str(type(child)):
				self.remove_attributes(child)
			else:
				try:
					del child.x
				except AttributeError:
					pass
				try:
					del child.y
				except AttributeError:
					pass

	def gradient(self,out_idx,x):
		self.model.zero_grad()
		use_keys=['X']
		for key in x.keys():
			if key not in use_keys:
				continue
			x[key].requires_grad_()
						
		outputs=self.model(x)
		if self.num_outputs==1:
			selected=[val for val in outputs]
		else:
			selected=[val for val in outputs[:,out_idx]]

		grads={}
		gkey_count=0
		for key in x.keys():
			if key not in use_keys:
				continue
			gkey_count+=1
			
			grad=torch.autograd.grad(selected,x[key],
								   retain_graph=True if gkey_count<len(use_keys) else None,
								   allow_unused=True)[0]
			if grad is not None:
				grads[key]=grad.cpu().numpy()
			else:
				grads[key]=torch.zeros_like(x[key]).cpu().numpy()
		return grads

	def shap_values(self,X):
		handles=self.add_handles(self.model,add_interim_values,deeplift_grad)
		output_phis=[]
		for i in range(self.num_outputs):
			phis=[]
			for index,x in enumerate(X):
				reads_num=x['X'].shape[-2]
				t_phis=torch.empty(reads_num).to(self.device)
				for j in range(reads_num):
					tiled_X={}					
					tiled_X['X']=x['X'][j:j+1,:].repeat(self.reads_mean['X'].shape[-2],1)
					tiled_X['kmer']=x['kmer'][j:j+1,:].repeat(x['kmer'].shape[-2],1)
					
					joint_X={}
					joint_X['X']=torch.cat((tiled_X['X'].unsqueeze(0),self.reads_mean['X'].unsqueeze(0)),dim=0)
					joint_X['kmer']=torch.cat((tiled_X['kmer'].unsqueeze(0),x['kmer'].unsqueeze(0)),dim=0)

					sample_phis=self.gradient(i,joint_X)
					t_sum=0
					for key in sample_phis.keys():
						if key in ['X']:
							s_phis=torch.from_numpy(sample_phis[key][1]).to(self.device)
							dif=x[key][j:j+1]-self.reads_mean[key]
							t_sum+=(s_phis*dif).mean(0).mean(-1)
					t_phis[j]=t_sum
				t_phis=t_phis.cpu().detach().numpy()
				phis.append(t_phis)
			output_phis.append(phis)
		for handle in handles:
			handle.remove()
		self.remove_attributes(self.model)
		return output_phis

	def interaction_gradient(self,output_idx,read_idx,x):
		self.model.zero_grad()
		use_keys=['X']
		for key in x.keys():
			if key not in use_keys:
				continue
			x[key].requires_grad_()
		output=self.model(x)

		grad_i=torch.autograd.grad(
			outputs=output,
			inputs=x['X'],
            grad_outputs=torch.ones_like(output),
			create_graph=True,
			retain_graph=True
		)[0]

		grad_i=grad_i.mean(dim=-1)

		batch_size,num_reads=grad_i.shape
		
		j=read_idx
		grad_i_j=grad_i[:,j].sum()
		grad_ij_j=torch.autograd.grad(
			outputs=grad_i_j,
			inputs=x['X'],
			retain_graph=True
		)[0]

		grad_ij_j=grad_ij_j.mean(dim=-1)

		with torch.no_grad():
			del grad_i, grad_i_j
		torch.cuda.empty_cache()

		return grad_ij_j

	def gradient_interaction_values(self,X):
		handles=self.add_handles(self.model,add_interim_values,deeplift_grad)
		output_interactions=[]
		for output_idx in range(self.num_outputs):
			per_output_interactions=[]
			for index,x in enumerate(X):
				reads_interactions=[]
				reads_num=x['X'].shape[-2]
				for i in range(reads_num):
					tiled_X={
						'X': x['X'][i:i+1,:].repeat(self.reads_mean['X'].shape[-2],1),
						'kmer': x['kmer'][i:i+1,:].repeat(x['kmer'].shape[-2],1)
					}
					joint_X={
						'X': torch.cat((tiled_X['X'].unsqueeze(0),self.reads_mean['X'].unsqueeze(0)),dim=0),
						'kmer': torch.cat((tiled_X['kmer'].unsqueeze(0),x['kmer'].unsqueeze(0)),dim=0)
					}
					interaction_grads=self.interaction_gradient(output_idx,i,joint_X)
					s_phis=interaction_grads[1].to(self.device)
					dif=x['X'][i:i+1]-self.reads_mean['X']
					dif=dif.mean(dim=-1)
					reads_interaction=s_phis*dif
					reads_interaction=reads_interaction.cpu().detach().numpy()
					reads_interactions.append(reads_interaction)
				per_output_interactions.append(reads_interactions)
			output_interactions.append(per_output_interactions)

		for handle in handles:
			handle.remove()
		self.remove_attributes(self.model)
		return np.array(output_interactions)

def deeplift_grad(module,grad_input,grad_output):
	module_type=module.__class__.__name__
	if module_type in op_handler:
		if op_handler[module_type].__name__ in ['nonlinear_1d']:
			return op_handler[module_type](module,grad_input,grad_output)
	return grad_input

def add_interim_values(module,input,output):
	try:
		del module.x
	except AttributeError:
		pass
	try:
		del module.y
	except AttributeError:
		pass
	module_type = module.__class__.__name__
	if module_type in op_handler:
		func_name = op_handler[module_type].__name__
		if func_name in ['nonlinear_1d']:
			if type(input) is tuple:
				setattr(module,'x',torch.nn.Parameter(input[0].detach()))
			else:
				setattr(module,'x',torch.nn.Parameter(input.detach()))
			if type(output) is tuple:
				setattr(module,'y',torch.nn.Parameter(output[0].detach()))
			else:
				setattr(module,'y',torch.nn.Parameter(output.detach()))

def passthrough(module,grad_input,grad_output):
	return None
def linear_1d(module,grad_input,grad_output):
	return None

def nonlinear_1d(module,grad_input,grad_output):
	#torch.Size([2, 2, 8, 1024]) torch.Size([2, 2, 8, 1024])
	delta_out=module.y[:int(module.y.shape[0]/2)]-module.y[int(module.y.shape[0]/2):]
	delta_in=module.x[:int(module.x.shape[0]/2)]-module.x[int(module.x.shape[0]/2):]

	dup0=[2]+[1 for i in delta_in.shape[1:]]
	grads=[None for _ in grad_input]
	grads[0]=torch.where(torch.abs(delta_in.repeat(dup0))<1e-6,grad_input[0],
						   grad_output[0]*(delta_out/delta_in).repeat(dup0))
	return tuple(grads)


op_handler = {}

op_handler['Dropout3d']=passthrough
op_handler['Dropout2d']=passthrough
op_handler['Dropout']=passthrough
op_handler['AlphaDropout']=passthrough

op_handler['Conv1d']=linear_1d
op_handler['Conv2d']=linear_1d
op_handler['Conv3d']=linear_1d
op_handler['ConvTranspose1d']=linear_1d
op_handler['ConvTranspose2d']=linear_1d
op_handler['ConvTranspose3d']=linear_1d
op_handler['Linear']=linear_1d
op_handler['AvgPool1d']=linear_1d
op_handler['AvgPool2d']=linear_1d
op_handler['AvgPool3d']=linear_1d
op_handler['AdaptiveAvgPool1d']=linear_1d
op_handler['AdaptiveAvgPool2d']=linear_1d
op_handler['AdaptiveAvgPool3d']=linear_1d
op_handler['BatchNorm1d']=linear_1d
op_handler['BatchNorm2d']=linear_1d
op_handler['BatchNorm3d']=linear_1d
##
op_handler['LayerNorm']=linear_1d

op_handler['LeakyReLU']=nonlinear_1d
op_handler['ReLU']=nonlinear_1d
op_handler['ELU']=nonlinear_1d
op_handler['Sigmoid']=nonlinear_1d
op_handler["Tanh"]=nonlinear_1d
op_handler["Softplus"]=nonlinear_1d
op_handler['Softmax']=nonlinear_1d
