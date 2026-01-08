import argparse
from dataset import*
from model import*

def test(model,test_loader,device,seq_reduce=0,read_reduce=0):
	model.eval()
	right_count,all_count=0,0
	prob_all,Y_all=[],[]
	with torch.no_grad():
		for _,l_dic in enumerate(test_loader):
			l_dic={k:v.to(device) for k, v in l_dic.items()}
			data_y=l_dic['label'].to(torch.int64)
			if seq_reduce==0:
				seq_feature=l_dic['seq_feature']
				seq_mask=l_dic['seq_mask']
				nano_feature=l_dic['nano_feature'][:,read_reduce:]
				nano_mask=l_dic['nano_mask'][:,read_reduce:]
			else:
				side_reduce=int(seq_reduce/2)
				seq_feature=l_dic['seq_feature'][:,side_reduce:-side_reduce]
				seq_mask=l_dic['seq_mask'][:,side_reduce:-side_reduce]
				nano_feature=l_dic['nano_feature'][:,read_reduce:,side_reduce:-side_reduce]
				nano_mask=l_dic['nano_mask'][:,read_reduce:,side_reduce:-side_reduce]
			pre_y=model(seq_feature,nano_feature,seq_mask,nano_mask)
			out_y=pre_y>0.5
			right_count+=out_y.eq(data_y).sum()
			all_count+=len(data_y)
			for each in pre_y:
				prob_all.append(np.array(each.cpu()))
			for each in data_y:
				Y_all.append(np.array(each.cpu()))
	roauc=roc_auc_score(Y_all,prob_all)

	accuracy=100*(right_count/all_count).item()
	print('AUC:{:.4f}, accuracy:{:.4f}%'.format(roauc,accuracy))
	torch.cuda.empty_cache()

def train(model,train_loader,val_loader,device,optimizer,loss_func,epochs,seq_reduce=0,read_reduce=0):
	torch.cuda.empty_cache()
	for epoch in range(epochs):
		total_loss=0
		model.train()
		for _,l_dic in enumerate(train_loader):
			l_dic={k:v.to(device) for k, v in l_dic.items()}
			data_y=l_dic['label']
			if seq_reduce==0:
				seq_feature=l_dic['seq_feature']
				seq_mask=l_dic['seq_mask']
				nano_feature=l_dic['nano_feature'][:,read_reduce:]
				nano_mask=l_dic['nano_mask'][:,read_reduce:]
			else:
				side_reduce=int(seq_reduce/2)
				seq_feature=l_dic['seq_feature'][:,side_reduce:-side_reduce]
				seq_mask=l_dic['seq_mask'][:,side_reduce:-side_reduce]
				nano_feature=l_dic['nano_feature'][:,read_reduce:,side_reduce:-side_reduce]
				nano_mask=l_dic['nano_mask'][:,read_reduce:,side_reduce:-side_reduce]
			pre_y=model(seq_feature,nano_feature,seq_mask,nano_mask)
			loss=loss_func(pre_y,data_y.float())
			total_loss+=loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		print('epoch {}, loss:{:.4f}'.format(epoch+1,total_loss.item()/len(train_loader)))
		if epoch%10==9:
			print('At epoch '+str(epoch+1),':')
			test(model,val_loader,device,seq_reduce,read_reduce)
			torch.save(model.state_dict(),'./model/model_'+str(epoch+1)+'_'+str(int(time.time()))+'.pkl')

if __name__ == '__main__':
	parser=argparse.ArgumentParser(description="Training")
	parser.add_argument('-l','--load_dataset_name',required=True,help="The name of saved dataset")
	parser.add_argument('-e','--epochs',default=150,type=int,help="Training epochs")
	parser.add_argument('-lr','--learning_rate',default=0.001,type=float,help="Learning rate")
	parser.add_argument('-sr','--seq_reduce',default=16,type=int,help="The number of not used sites")
	parser.add_argument('-rr','--read_reduce',default=0,type=int,help="The number of not used reads")

	args=parser.parse_args()


	with open('../edata/Save_DataSet/'+args.load_dataset_name+'_train_set.pkl','rb') as f:
		flattened_train_set=pickle.load(f)
	with open('../edata/Save_DataSet/'+args.load_dataset_name+'_val_set.pkl','rb') as f:
		flattened_val_set=pickle.load(f)
	print('len(flattened_train_set)',len(flattened_train_set))
	print('len(flattened_val_set)',len(flattened_val_set))
	
	train_loader=DataLoader(flattened_train_set,batch_size=5,shuffle=True,drop_last=True)
	val_loader=DataLoader(flattened_val_set,batch_size=5,shuffle=True)
	
	torch.cuda.manual_seed_all(0)
	
	device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
	model=Nano(c_s=4,c_x=3,c_emb=16,c_hidden_att=16,c_o=1,no_heads=8,blocks_lis=[2,2,2,0,0,0],
				dropout=0.2,transition_n=2,inf=1e9,eps=1e-8,
				clear_cache_between_blocks=False).to(device)
	optimizer=optim.Adam(model.parameters(),lr=args.learning_rate)
	loss_func=nn.BCELoss()
	train(model,train_loader,val_loader,device,optimizer,loss_func,args.epochs,seq_reduce=args.seq_reduce,read_reduce=args.read_reduce)
