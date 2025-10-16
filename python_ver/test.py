import argparse
from dataset import*
from model import*

def detailed_test(model,test_loader,device,seq_reduce=0,read_reduce=0,curve_name=None,histo_name=None):
	model.eval()
	right_count,all_count=0,0
	more_dict={0.5:[0,0],0.6:[0,0],0.8:[0,0],0.7:[0,0],0.9:[0,0],0.95:[0,0],0.98:[0,0],\
			   0.99:[0,0],0.995:[0,0],0.999:[0,0],0.9995:[0,0],0.9999:[0,0],0.99995:[0,0],\
			   0.99999:[0,0],0.999995:[0,0],0.999999:[0,0]}
	prob_all,Y_all=[],[]
	motif_dict={}
	range_list=[]
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
			for key in more_dict:
				more_dict[key][0]+=((pre_y>key)&data_y).sum()
				more_dict[key][1]+=(pre_y>key).sum()

			if histo_name:
				middle_pos=int((len(l_dic['seq_feature'][0])-1)/2)
				center_seqs=l_dic['seq_feature'][:,middle_pos-2:middle_pos+3]			
				for i in range(len(bdata_yy)):
					_Seq=''
					for j in range(5):
						if abs(center_seqs[i][j][0]-1)<0.01:
							_Seq+='A'
						elif abs(center_seqs[i][j][1]-1)<0.01:
							_Seq+='T'
						elif abs(center_seqs[i][j][2]-1)<0.01:
							_Seq+='C'
						elif abs(center_seqs[i][j][3]-1)<0.01:
							_Seq+='G'
						else:
							_Seq+='N'
					if 'N' not in _Seq:
						if _Seq not in motif_dict:
							motif_dict[_Seq]={'TP':0,'FP':0,'TN':0,'FN':0}
						if out_y[i]==1 and data_y[i]==1:
							motif_dict[_Seq]['TP']+=1
						elif out_y[i]==1 and data_y[i]==0:
							motif_dict[_Seq]['FP']+=1
						elif out_y[i]==0 and data_y[i]==0:
							motif_dict[_Seq]['TN']+=1
						elif out_y[i]==0 and data_y[i]==1:
							motif_dict[_Seq]['FN']+=1
				for i in range(len(ry)):
					range_list.append([ry[i].cpu().item(),data_y[i].cpu().item()])
	if histo_name:
		save_frame=pd.DataFrame(motif_dict).T
		save_frame.to_csv('./edata/Save_for_drawing/'+histo_name+'_motif_histo.csv',index=True,sep=',')
		save_frame=pd.DataFrame(range_list)
		save_frame.columns=['Probability score','Ground Truth']
		save_frame.to_csv('./edata/Save_for_drawing/'+histo_name+'_range_histo.csv',index=False,sep=',')
	if curve_name:
		save_frame=pd.DataFrame({'label':Y_all,'pred':prob_all})
		save_frame.to_csv('./edata/Save_for_drawing/'+curve_name+'_curve.csv',index=False,sep=',')

	print('Im total',all_count,'samples:')
	auc=roc_auc_score(Y_all,prob_all)
	accuracy=100*(right_count/all_count).item()
	print('AUC:{:.4f}   accuracy:{:.4f}%'.format(auc,accuracy))
	for key in more_dict:
		if more_dict[key][1]>0:
			print('Precision when positive threshold at {:g} is :{:.4f}% (total:{:d})'.format(key,100*more_dict[key][0]/more_dict[key][1],more_dict[key][1]))
	torch.cuda.empty_cache()


if __name__ == '__main__':
	parser=argparse.ArgumentParser(description="Training")
	parser.add_argument('-l','--load_dataset_name',required=True,help="The name of saved dataset, should be in the folder 'edata/Save_DataSet'")
	parser.add_argument('-m','--load_model_name',required=True,help="The loaded model for testing, should be in the folder 'model'")
	parser.add_argument('-sr','--seq_reduce',default=16,type=int,help="The number of not used sites")
	parser.add_argument('-rr','--read_reduce',default=0,type=int,help="The number of not used reads")

	args=parser.parse_args()

	with open('../edata/Save_DataSet/'+args.load_dataset_name+'_test_set.pkl','rb') as f:
		flattened_test_set=pickle.load(f)
	print('len(flattened_test_set)',len(flattened_test_set))
	test_loader=DataLoader(flattened_test_set,batch_size=5,shuffle=True)

	device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model=Nano(c_s=4,c_x=3,c_emb=16,c_hidden_att=16,c_o=1,no_heads=8,blocks_lis=[2,2,2,0,0,0],
				dropout=0.2,transition_n=2,inf=1e9,eps=1e-8,
				clear_cache_between_blocks=False).to(device)
	#'../model/NSWord_222000_50_50reads_9sites.pkl'
	model.load_state_dict(torch.load('../model/'+args.load_model_name+'.pkl',weights_only=True))
	detailed_test(model,test_loader,device,seq_reduce=args.seq_reduce,read_reduce=args.read_reduce,curve_name=None,histo_name=None)
