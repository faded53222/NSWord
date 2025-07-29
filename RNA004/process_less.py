import argparse
import os
import json
import time
import random
import multiprocessing
import pandas as pd
import numpy as np
from io import StringIO
from itertools import groupby
from operator import itemgetter

import parallels

SIDE_WINDOW_SIZE,MIN_COVER_RATE=12,0.6

def search_select_info(read_info,read_index,select_info):
	read_info=read_info[read_info['read_index']==read_index].copy()
	read_info=read_info[read_info['reference_kmer']==read_info['model_kmer']].copy()
	read_info=read_info[(read_info['position']>=select_info[1]-SIDE_WINDOW_SIZE)&(read_info['position']<=select_info[1]+SIDE_WINDOW_SIZE)].copy()	

	if len(read_info.index)==0:
		return -1,-1
	if len(read_info[read_info['position']==select_info[1]].index)==0:
		return -1,-1
	if len(np.unique(list(read_info['position'])))<int((1+2*SIDE_WINDOW_SIZE)*MIN_COVER_RATE):
		return -1,-1

	read_info['length']=(read_info['end_idx']-read_info['start_idx']).astype(np.int32)
	read_info['sum_level_mean']=read_info['event_level_mean'].astype(np.float64)*read_info['length']
	read_info['sum_stdv']=read_info['event_stdv'].astype(np.float64)*read_info['length']
	read_info['sum_dwell_time']=read_info['event_length'].astype(np.float64)*read_info['length']

	group_keys=['contig','read_index','position','reference_kmer']
	read_info=read_info.groupby(group_keys)

	total_length=read_info['length'].sum()
	sum_level_mean=read_info['sum_level_mean'].sum()
	sum_stdv=read_info['sum_stdv'].sum()
	sum_dwell_time=read_info['sum_dwell_time'].sum()
	
	read_info=pd.concat([read_info['start_idx'].min(),read_info['end_idx'].max()],axis=1)
	read_info['level_mean']=round(sum_level_mean/total_length,6)
	read_info['stdv']=round(sum_stdv/total_length,6)
	read_info['dwell_time']=round(sum_dwell_time/total_length,6)
	read_info=read_info.reset_index()
	read_info=read_info.loc[np.argsort(read_info['position'])].reset_index(drop=True)


	partition_str=''
	partition_data=[]

	pos_start=select_info[1]-SIDE_WINDOW_SIZE-1
	pos_end=select_info[1]+SIDE_WINDOW_SIZE+1
	c_pos=pos_start
	
	for j in range(len(read_info.index)):
		j_pos=read_info.iloc[j]['position']
		for k in reversed(range(j_pos-c_pos-1)):
			if k<2:
				partition_str+=read_info.iloc[j]['reference_kmer'][1-k]
			else:
				partition_str+='N'
			partition_data.append([-1,-1,-1])
		partition_str+=read_info.iloc[j]['reference_kmer'][2]#2 is mid of 5-mer
		partition_data.append([read_info.iloc[j]['level_mean'],read_info.iloc[j]['stdv'],read_info.iloc[j]['dwell_time']])
		c_pos=j_pos

	for k in range(pos_end-c_pos-1):
		if k<2:
			partition_str+=read_info.iloc[-1]['reference_kmer'][3+k]
		else:
			partition_str+='N'
		partition_data.append([-1,-1,-1])
	return partition_str,partition_data

def search_and_partition(file,o_file,gene,gene_readindex_rows,search_info,num_of_reads_per_bag):
	partitions_dic={}
	for _,index_info in gene_readindex_rows:
		with open(file+'.txt','r') as f:
			f.seek(index_info['file_pos_start'],0)
			read_str=f.read(index_info['file_pos_end']-index_info['file_pos_start'])
		read_index=index_info['read_index']
		read_info=pd.read_csv(StringIO(read_str),delimiter='\t',index_col=False,
						names=['contig','position','reference_kmer','read_index','strand','event_index',
							   'event_level_mean','event_stdv','event_length','model_kmer','model_mean',
							   'model_stdv','standardized_level','start_idx','end_idx'])
		for select_info in search_info:
			partition_str,partition_data=search_select_info(read_info,read_index,select_info)
			if partition_str!=-1:
				if select_info not in partitions_dic:
					partitions_dic[select_info]=[]
				partitions_dic[select_info].append((partition_str,partition_data))
	
	for key in partitions_dic:
		repeats=partitions_dic[key]
		with open(o_file+'.json','a') as fw_json, open(o_file+'.index','a') as fw_json_index:
			file_pos_start=fw_json.tell()
			combine_seq=list(repeats[0][0])
			for i in range(len(combine_seq)):
				if combine_seq[i]=='N':
					for j in range(len(repeats)):
						if repeats[j][0][i]!='N':
							combine_seq[i]=repeats[j][0][i]
							break
			combine_seq=''.join(combine_seq)
			
			into_k_bags=len(repeats)//num_of_reads_per_bag
			for bag_number in range(0,into_k_bags):
				fw_json.write(json.dumps(combine_seq)+'\n')
				for repeat in repeats[bag_number*num_of_reads_per_bag:(bag_number+1)*num_of_reads_per_bag]:
					fw_json.write(json.dumps(repeat[1])+'\n')
				file_pos_end=fw_json.tell()
				fw_json_index.write('%s_%s_%d\t%d\t%d\n'%(gene,key[0],key[1],file_pos_start,file_pos_end))

def process(file,restrict_file,num_of_reads_per_bag):
	o_file=file+'__'+restrict_file.split('/')[-1]
	open(o_file+'.json','w')
	open(o_file+'.index','w')
	
	restrict_gene_dic={}
	with open(restrict_file+'.txt','r') as f:
		for line in f.readlines():
			items=line.strip().split('_')
			if items[0] not in restrict_gene_dic:
				restrict_gene_dic[items[0]]=[]
			restrict_gene_dic[items[0]].append((items[1],int(items[2])))

	index_file=pd.read_csv(file+'.index')
	genes=list(dict.fromkeys(index_file['contig'].values.tolist()))
	index_file=index_file.set_index('contig')

	for gene in genes:
		read_index_l_info=list(index_file.loc[[gene]].iterrows())
		search_info=restrict_gene_dic[gene.split('_')[-1]]
		print(gene,len(read_index_l_info))
		search_and_partition(file,o_file,gene,read_index_l_info,search_info,num_of_reads_per_bag)

if __name__ == '__main__':
	parser=argparse.ArgumentParser(description="Get desired information from nanopolish events")
	parser.add_argument('-i','--input',required=True,help="Input events file path")
	parser.add_argument('-r','--restrict_file',required=True,help="DRACH motifs to get from the input file")
	parser.add_argument('-n','--num_of_reads_per_bag',required=True,help="As there are few different sequences but more reads per sequence,\
						also there are more likely modified sites in m6A-runs, we can put less reads into a bag to make more bags.")
	args=parser.parse_args()

	print('begin the processing of',args.input.split('/')[-1])
	process(args.input,args.restrict_file,int(args.num_of_reads_per_bag))
	print('end the processing of',args.input.split('/')[-1])
