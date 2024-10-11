from imports import *

Seq_Coding={'A':[1.,0.,0.,0.],'T':[0.,1.,0.,0.],'C':[0.,0.,1.,0.],'G':[0.,0.,0.,1.],'N':[0.25,0.25,0.25,0.25]}
class NanoDataset(torch.utils.data.Dataset):
	def __init__(self,path,use_file):
		samples_dic={}
		with open(path+'/'+use_file+'.txt','r') as f:
			for line in f.readlines():
				f_name,label=line.strip().split()
				with open(path+'/'+f_name+'.index') as f2:
					for line2 in f2.readlines():
						sample,start,end=line2.strip().split('\t')
						if sample not in samples_dic:
							samples_dic[sample]=[]
						samples_dic[sample].append((f_name,int(start),int(end),int(label)))
		self.path=path
		self.samples_keys=list(samples_dic.keys())
		self.samples_dic=samples_dic

	def __getitem__(self,index):
		R_dicts=[]
		for single_sample in self.samples_dic[self.samples_keys[index]]:
			file,seek_start,seek_end,label=single_sample
			R_dict={'seq_feature':[],'seq_mask':[],'nano_feature':[],'nano_mask':[],'label':label}
			with open(self.path+'/'+file+'.json') as f:
				f.seek(seek_start,0)
				json_str=f.read(seek_end-seek_start)
				Ls=json_str.strip().split('\n')
				for each in json.loads(Ls[0]):
					R_dict['seq_feature'].append(Seq_Coding[each])
					if each=='N':
						R_dict['seq_mask'].append(0)
					else:
						R_dict['seq_mask'].append(1)

				for L in Ls[1:]:
					L_data=json.loads(L)
					t_feature=[]
					t_mask=[]
					for each in L_data:
						if each[0]<0:
							t_feature.append([0,0,0])
							t_mask.append(0)
						else:
							t_feature.append(each)
							t_mask.append(1)
					R_dict['nano_mask'].append(t_mask)
					R_dict['nano_feature'].append(t_feature)
			for key in R_dict:
				R_dict[key]=torch.tensor(R_dict[key])
			R_dicts.append(R_dict)
		return R_dicts				
	def __len__(self):
		return len(self.samples_dic)

class FlattenedDataset(torch.utils.data.Dataset):
	def __init__(self,original_dataset):
		self.flattened_data=[]
		for data in original_dataset:
			self.flattened_data.extend(data)
	def __len__(self):
		return len(self.flattened_data)
	def __getitem__(self,index):
		if isinstance(index,(list,np.ndarray)):
			return [self.flattened_data[i] for i in index]
		else:
			return self.flattened_data[index]
