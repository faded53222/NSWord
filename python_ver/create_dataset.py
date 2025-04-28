import argparse
from dataset import*

if __name__ == '__main__':
	parser=argparse.ArgumentParser(description="Create and save dataset")
	parser.add_argument('-p','--path',required=True,help="Dataset path, should be in the folder 'edata/DataSet'")
	parser.add_argument('-u','--use_file_name',required=True,help="A file in previous path that records the files required to construct the dataset, where each line follows the format: 'filename (tab) label (0/1)'")
	parser.add_argument('-s','--save_name',required=True,help="The name for saved dataset, will be saved to the folder 'edata/Save_DataSet'")

	args=parser.parse_args()

	dataset=NanoDataset('../edata/DataSet/'+args.path,args.use_file_name)
    train_size=int(len(dataset)*0.6)
    val_size=int(len(dataset)*0.2)
    test_size=len(dataset)-val_size-train_size
    train_set,val_set,test_set=torch.utils.data.random_split(dataset,[train_size,val_size,test_size])
	flattened_train_set=FlattenedDataset(train_set)
	flattened_val_set=FlattenedDataset(val_set)
	flattened_test_set=FlattenedDataset(test_set)
	print('len(flattened_train_set)',len(flattened_train_set))
	print('len(flattened_val_set)',len(flattened_val_set))
	print('len(flattened_test_set)',len(flattened_test_set))

	with open('../edata/Save_DataSet/'+args.save_name+'_train_set.pkl','wb') as f:
		pickle.dump(flattened_train_set,f)
	with open('../edata/Save_DataSet/'+args.save_name+'_val_set.pkl','wb') as f:
		pickle.dump(flattened_val_set,f)
	with open('../edata/Save_DataSet/'+args.save_name+'_test_set.pkl','wb') as f:
		pickle.dump(flattened_test_set,f)
