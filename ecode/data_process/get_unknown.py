import argparse
import json
import random
import multiprocessing
import pandas as pd
import numpy as np
from io import StringIO
from itertools import groupby

import parallels

def parallel_unknown_seq_search(read_info,search_seqs,avoid_poses,child_conn,locks):
	searched_pos=[]
	parallel_seq_search_dic={}
	for seq in search_seqs:
		parallel_seq_search_dic[seq]=[]
		
	read_info=read_info[read_info['reference_kmer']==read_info['model_kmer']].copy()	
	if len(read_info.index)==0:
		child_conn.send(None)
		return
	for _,row in read_info.iterrows():
		if row['position'] in searched_pos:
			continue
		searched_pos.append(row['position'])
		if row['position'] not in avoid_poses:
			if row['reference_kmer'] in search_seqs:
				parallel_seq_search_dic[row['reference_kmer']].append(row['position'])

	with locks['pipe_1']:
		child_conn.send(parallel_seq_search_dic)

def receive_unknown_seq_search(recv_num,search_seqs,r_child_conn,parent_conn):
	seq_search_dic={}
	for seq in search_seqs:
		seq_search_dic[seq]={}
	for i in range(recv_num):
		while 1:
			try:
				parallel_seq_search_dic=parent_conn.recv()
				if parallel_seq_search_dic is not None:
					for seq in search_seqs:
						for pos in parallel_seq_search_dic[seq]:
							if pos not in seq_search_dic[seq]:
								seq_search_dic[seq][pos]=0
							seq_search_dic[seq][pos]+=1
				break
			except EOFError:
				sleep(0.01)

	max_select_keep=(0,0)
	max_count_keep=0
	for seq in seq_search_dic:
		for pos in seq_search_dic[seq]:
			if seq_search_dic[seq][pos]>max_count_keep:
				max_count_keep=seq_search_dic[seq][pos]
				max_select_keep=(seq,pos)
	if max_count_keep>REPEAT_SIZE_MIN:
		r_child_conn.send(max_select_keep)
	else:
		r_child_conn.send(None)

def parallel_unknown_get_partition(read_info,select_info,child_conn,locks):
	read_info=read_info[read_info['reference_kmer']==read_info['model_kmer']].copy()
	read_info=read_info[(read_info['position']>=select_info[1]-SIDE_WINDOW_SIZE)&(read_info['position']<=select_info[1]+SIDE_WINDOW_SIZE)].copy()
	if len(read_info.index)==0:
		child_conn.send(None)
		return
	if len(np.unique(list(read_info['position'])))<int((1+2*SIDE_WINDOW_SIZE)*MIN_COVER_RATE):
		child_conn.send(None)
		return

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

	pos_start=select_info[1]-SIDE_WINDOW_SIZE-1
	pos_end=select_info[1]+SIDE_WINDOW_SIZE+1

	partition_str=''
	partition_data=[]
	c_pos=pos_start

	for j in range(len(read_info.index)):
		j_pos=read_info.iloc[j]['position']
		for k in reversed(range(j_pos-c_pos-1)):
			if k<2:
				partition_str+=read_info.iloc[j]['reference_kmer'][1-k]
			else:
				partition_str+='N'
			partition_data.append([-1,-1,-1])
		partition_str+=read_info.iloc[j]['reference_kmer'][2]
		partition_data.append([read_info.iloc[j]['level_mean'],read_info.iloc[j]['stdv'],read_info.iloc[j]['dwell_time']])
		c_pos=j_pos

	for k in range(pos_end-c_pos-1):
		if k<2:
			partition_str+=read_info.iloc[-1]['reference_kmer'][3+k]
		else:
			partition_str+='N'
		partition_data.append([-1,-1,-1])

	with locks['pipe_2']:
		child_conn.send((partition_str,partition_data))

def receive_unknown_partition(recv_num,gene,r_child_conn,parent_conn):
	recv_list=[]
	for i in range(recv_num):
		while 1:
			try:
				recv_pdata=parent_conn.recv()
				if recv_pdata is not None:
					recv_list.append(recv_pdata)
				break
			except EOFError:
				sleep(0.01)
	
	if len(recv_list)<REPEAT_SIZE_MIN:
		r_child_conn.send(None)
	else:
		if len(recv_list)>REPEAT_SIZE_MAX:
			recv_list=recv_list[:REPEAT_SIZE_MAX]
		combine_seq=list(recv_list[0][0])
		for i in range(len(combine_seq)):
			if combine_seq[i]=='N':
				for j in range(len(recv_list)):
					if recv_list[j][0][i]!='N':
						combine_seq[i]=recv_list[j][0][i]
						break
		combine_seq=''.join(combine_seq)
		r_child_conn.send((combine_seq,[x[1] for x in recv_list]))

def parallel_unknown_main(file,gene,gene_readindex_rows,needed_dict,avoid_poses,task_queue_1,task_queue_2,child_conn,locks):	
	search_seqs=[]
	for key in needed_dict.keys():
		search_seqs.append(key)

	parent_conn_1,child_conn_1=multiprocessing.Pipe()
	r_parent_conn_1,r_child_conn_1=multiprocessing.Pipe()

	recv_num_1=len(gene_readindex_rows)
	receiver_1=multiprocessing.Process(target=receive_unknown_seq_search,args=(recv_num_1,search_seqs,r_child_conn_1,parent_conn_1))
	receiver_1.start()

	for _,index_info in gene_readindex_rows:
		with locks['read'],open(file+'.txt','r') as f:
			f.seek(index_info['file_pos_start'],0)
			read_str=f.read(index_info['file_pos_end']-index_info['file_pos_start'])
		read_info=pd.read_csv(StringIO(read_str),delimiter='\t',index_col=False,
						names=['contig','position','reference_kmer','read_index','strand','event_index',
							   'event_level_mean','event_stdv','event_length','model_kmer','model_mean',
							   'model_stdv','standardized_level','start_idx','end_idx'])
		task_queue_1.put((read_info,search_seqs,avoid_poses,child_conn_1))
	receiver_1.join()
	select_info=r_parent_conn_1.recv()
	if select_info is None:
		child_conn.send((None,None,None))
		return
	
	recv_num_2=len(gene_readindex_rows)
	parent_conn_2,child_conn_2=multiprocessing.Pipe()
	r_parent_conn_2,r_child_conn_2=multiprocessing.Pipe()
	receiver_2=multiprocessing.Process(target=receive_unknown_partition,args=(recv_num_2,gene,r_child_conn_2,parent_conn_2))
	receiver_2.start()

	for _,index_info in gene_readindex_rows:
		with locks['read'],open(file+'.txt','r') as f:
			f.seek(index_info['file_pos_start'],0)
			read_str=f.read(index_info['file_pos_end']-index_info['file_pos_start'])
		read_info=pd.read_csv(StringIO(read_str),delimiter='\t',index_col=False,
						names=['contig','position','reference_kmer','read_index','strand','event_index',
							   'event_level_mean','event_stdv','event_length','model_kmer','model_mean',
							   'model_stdv','standardized_level','start_idx','end_idx'])
		task_queue_2.put((read_info,select_info,child_conn_2))

	receiver_2.join()
	partition_data=r_parent_conn_2.recv()
	with locks['pipe_0']:
		child_conn.send((gene,select_info,partition_data))

def receive_unknown_main(recv_num,file,restrict_file_name,needed_dict,r_child_conn,parent_conn):
	LABEL=0
	for i in range(recv_num):
		while 1:
			try:
				gene,select_info,partition_data=parent_conn.recv()
				if partition_data is not None:
					if select_info[0] in needed_dict:
						needed_dict[select_info[0]]-=1
						if needed_dict[select_info[0]]<=0:
							del needed_dict[select_info[0]]
						with open(file+'__'+restrict_file_name+'_unknown.json','a') as fw_json:
							with open(file+'__'+restrict_file_name+'_unknown.index','a') as fw_json_index:
								file_pos_start=fw_json.tell()
								fw_json.write(json.dumps(partition_data[0])+'\n')
								repeats=partition_data[1]
								for repeat in repeats:
									fw_json.write(json.dumps(repeat)+'\n')
								for i in range(REPEAT_SIZE_MAX-len(repeats)):
									rand_select=random.randint(0,len(repeats)-1)
									fw_json.write(json.dumps(repeats[rand_select])+'\n')
								file_pos_end=fw_json.tell()
								fw_json_index.write('%s_%s_%d\t%d\t%d\n'%(gene,select_info[0],select_info[1],file_pos_start,file_pos_end))
				break
			except EOFError:
				sleep(0.01)
	r_child_conn.send(needed_dict)

def parallel_process_unknown(file,restrict_file_name,n_processes_0,n_processes_1,n_processes_2,Get_All=False):
	lock_0=dict()
	for lock_type in ['read','pipe_0','pipe_1','pipe_2']:
		lock_0[lock_type]=multiprocessing.Lock()
	task_queue_0=multiprocessing.JoinableQueue(maxsize=n_processes_0*2)
	consumers_0=[parallels.Consumer(task_queue=task_queue_0,task_function=parallel_unknown_main,locks=lock_0) for i in range(n_processes_0)]
	for process_0 in consumers_0:
		process_0.start()
 
	open(file+'__'+restrict_file_name+'_unknown.json','w')
	open(file+'__'+restrict_file_name+'_unknown.index','w')

	manager=multiprocessing.Manager()
	task_queue_1=manager.JoinableQueue(maxsize=n_processes_1*2)
	consumers_1=[parallels.Consumer(task_queue=task_queue_1,task_function=parallel_unknown_seq_search,locks=lock_0) for i in range(n_processes_1)]
	for process_1 in consumers_1:
		process_1.start()

	task_queue_2=manager.JoinableQueue(maxsize=n_processes_2*2)
	consumers_2=[parallels.Consumer(task_queue=task_queue_2,task_function=parallel_unknown_get_partition,locks=lock_0) for i in range(n_processes_2)]
	for process_2 in consumers_2:
		process_2.start()

	needed_dict={}
	avoid_dict={}
	with open(file+'__'+restrict_file_name+'.index','r') as f:
		lines=f.readlines()
		for line in lines:
			sinfo=line.split()[0]
			gene,seq,pos=sinfo.split('_')
			if seq not in needed_dict.keys():
				needed_dict[seq]=0
			needed_dict[seq]+=1
			if gene not in avoid_dict:
				avoid_dict[gene]=[]
			avoid_dict[gene].append(int(pos))

	MAX=99999
	if Get_All:
		for key in needed_dict:
			needed_dict[key]=MAX
		print('now got: ',{key: MAX-value for key,value in needed_dict.items()})
	else:
		print('now need: ',needed_dict)
	
	index_file=pd.read_csv(file+'.index')
	genes=list(dict.fromkeys(index_file['transcript_id'].values.tolist()))
	random.shuffle(genes)
	index_file=index_file.set_index('transcript_id')

	parent_conn_0,child_conn_0=multiprocessing.Pipe()	
	i=0
	end_search=0
	while 1:
		actual_recv_num=0
		temp_to_send=[]
		while 1:
			if i>=len(genes):
				end_search=1
				break
			gene=genes[i]
			i+=1
			read_index_l_info=list(index_file.loc[[gene]].iterrows())
			if len(read_index_l_info)<50:
				continue
			print(gene,len(read_index_l_info))
			avoid_poses=[]
			if gene[:15] in avoid_dict:
				avoid_poses=avoid_dict[gene[:15]]
			temp_to_send.append((file,gene[:15],read_index_l_info[:500],needed_dict,avoid_poses,task_queue_1,task_queue_2,child_conn_0))
			actual_recv_num+=1
			if actual_recv_num>=RECV_NUM:
				break

		t_parent_conn,t_child_conn=multiprocessing.Pipe()
		t_receiver=multiprocessing.Process(target=receive_unknown_main,args=(actual_recv_num,file,restrict_file_name,needed_dict,t_child_conn,parent_conn_0))
		t_receiver.start()
		for task in temp_to_send:
			task_queue_0.put(task)
		t_receiver.join()
		needed_dict=t_parent_conn.recv()
		if Get_All:
			print('now got: ',{key: MAX-value for key,value in needed_dict.items()})
		else:
			print('now need: ',needed_dict)
		if len(needed_dict.keys())==0 or end_search==1:
			break

	for _ in range(n_processes_0):
		task_queue_0.put(None)
	for _ in range(n_processes_1):
		task_queue_1.put(None)
	for _ in range(n_processes_2):
		task_queue_2.put(None)

if __name__ == '__main__':
	parser=argparse.ArgumentParser(description="Get desired information of the same motifs of input file from nanopolish events, number not limited")
	parser.add_argument('-i','--input',required=True,help="Input file path")
	parser.add_argument('-r','--restrict_file',required=True,help="ENST motifs to get from the input file")

	parser.add_argument('-n0','--n_processes_0',default=3,help="The number of processes for processing task queue 0")
	parser.add_argument('-n1','--n_processes_1',default=6,help="The number of processes for processing task queue 1")
	parser.add_argument('-n2','--n_processes_2',default=12,help="The number of processes for processing task queue 2")

	parser.add_argument('-s','--side_window_size',default=12,help="The number of neighbor sites to obtain for each side")
	parser.add_argument('-cr','--min_cover_rate',default=0.8,help="The lowest rate of available sites for a segment to be qualified")
	parser.add_argument('-ri','--repeat_size_min',default=20,help="The lowest number of reads for a segment to be qualified")
	parser.add_argument('-ra','--repeat_size_max',default=50,help="The number of obtained reads")
	parser.add_argument('-rn','--recv_num',default=12,help="The number of tasks waiting to be completed at one time")

	args=parser.parse_args()

	global SIDE_WINDOW_SIZE,MIN_COVER_RATE,REPEAT_SIZE_MIN,REPEAT_SIZE_MAX,RECV_NUM
	SIDE_WINDOW_SIZE=args.side_window_size
	MIN_COVER_RATE=args.min_cover_rate
	REPEAT_SIZE_MIN=args.repeat_size_min
	REPEAT_SIZE_MAX=args.repeat_size_max
	RECV_NUM=args.recv_num

	print('begin the processing of',args.input.split('/')[-1])
	parallel_process_unknown(args.input,args.restrict_file,args.n_processes_0,args.n_processes_1,args.n_processes_2,Get_All=True)
	print('end the processing of',args.input.split('/')[-1])
