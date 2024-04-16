####change the file names to those you want to process in main.
####pay attention to the note in line 203.
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

N_processes_0=3
N_processes_1=6
N_processes_2=12
SIDE_WINDOW_SIZE=12
MIN_COVER_RATE=0.8
REPEAT_SIZE_MIN=20
REPEAT_SIZE_MAX=50


def parallel_seq_search(read_info,search_seqs,child_conn,locks):
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
        if row['reference_kmer'] in search_seqs:
            parallel_seq_search_dic[row['reference_kmer']].append(row['position'])

    with locks['pipe_1']:
        child_conn.send(parallel_seq_search_dic)

def receive_seq_search(recv_num,search_seqs,r_child_conn,parent_conn):
    seq_search_dic={}
    for seq in search_seqs:
        seq_search_dic[seq]=[]
    for i in range(recv_num):
        while 1:
            try:
                parallel_seq_search_dic=parent_conn.recv()
                if parallel_seq_search_dic is not None:
                    for seq in search_seqs:
                        for pos in parallel_seq_search_dic[seq]:
                            if pos not in seq_search_dic[seq]:
                                seq_search_dic[seq].append(pos)
                break
            except EOFError:
                sleep(0.01)
    r_child_conn.send(seq_search_dic)

def parallel_get_partition(read_info,select_info,child_conn,locks):
    read_info=read_info[read_info['reference_kmer']==read_info['model_kmer']].copy()
    read_info=read_info[(read_info['position']>=select_info[1]-SIDE_WINDOW_SIZE)&(read_info['position']<=select_info[1]+SIDE_WINDOW_SIZE)].copy()
    if len(read_info.index)==0:
        child_conn.send((select_info,None))
        return
    if len(np.unique(list(read_info['position'])))<int((1+2*SIDE_WINDOW_SIZE)*MIN_COVER_RATE):
        child_conn.send((select_info,None))
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
    read_info['level_mean']=sum_level_mean/total_length
    read_info['stdv']=sum_stdv/total_length
    read_info['dwell_time']=sum_dwell_time/total_length
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
        child_conn.send((select_info,(partition_str,partition_data)))

def receive_partition(recv_num,file,gene,all_select_info,parent_conn,locks):
    LABEL=1
    partitions_dic={}
    for select_info in all_select_info:
        partitions_dic[select_info]=[]
    for i in range(recv_num):
        while 1:
            try:
                recv_select_info,recv_pdata=parent_conn.recv()
                if recv_pdata is not None:
                    partitions_dic[recv_select_info].append(recv_pdata)
                break
            except EOFError:
                sleep(0.01)
    
    for key in partitions_dic:
        repeats=partitions_dic[key]
        if len(repeats)>REPEAT_SIZE_MIN:
            if len(repeats)>REPEAT_SIZE_MAX:
                repeats=repeats[:REPEAT_SIZE_MAX]
            with locks['write'],open(file+'_processed.json','a') as fw_json:
                with open(file+'_processed.index','a') as fw_json_index:
                    file_pos_start=fw_json.tell()
                    combine_seq=list(repeats[0][0])
                    for i in range(len(combine_seq)):
                        if combine_seq[i]=='N':
                            for j in range(len(repeats)):
                                if repeats[j][0][i]!='N':
                                    combine_seq[i]=repeats[j][0][i]
                                    break
                    combine_seq=''.join(combine_seq)
                    fw_json.write(json.dumps(combine_seq)+'\n')
                    for repeat in repeats:
                        fw_json.write(json.dumps(repeat[1])+'\n')
                    for i in range(REPEAT_SIZE_MAX-len(repeats)):
                        rand_select=random.randint(0,len(repeats)-1)
                        fw_json.write(json.dumps(repeats[rand_select][1])+'\n')
                    file_pos_end=fw_json.tell()
                    fw_json_index.write('%s_%s_%d\t%d\t%d\t%d\n'%(gene,key[0],key[1],file_pos_start,file_pos_end,LABEL))

def parallel_main(file,gene,gene_readindex_rows,search_info,task_queue_1,task_queue_2,locks):
    parent_conn_1,child_conn_1=multiprocessing.Pipe()
    r_parent_conn_1,r_child_conn_1=multiprocessing.Pipe()

    recv_num_1=len(gene_readindex_rows)
    search_seqs=np.unique(list(map(lambda x:x[1],search_info)))        
    receiver_1=multiprocessing.Process(target=receive_seq_search,args=(recv_num_1,search_seqs,r_child_conn_1,parent_conn_1))
    receiver_1.start()

    for _,index_info in gene_readindex_rows:
        with locks['read'],open(file+'.txt','r') as f:
            f.seek(index_info['file_pos_start'],0)
            read_str=f.read(index_info['file_pos_end']-index_info['file_pos_start'])
        read_info=pd.read_csv(StringIO(read_str),delimiter='\t',index_col=False,
                        names=['contig','position','reference_kmer','read_index','strand','event_index',
                               'event_level_mean','event_stdv','event_length','model_kmer','model_mean',
                               'model_stdv','standardized_level','start_idx','end_idx'])
        task_queue_1.put((read_info,search_seqs,child_conn_1))
    receiver_1.join()
    seq_search_dic=r_parent_conn_1.recv()

    search_info_dic={}
    for each in search_info:
        if each[1] not in search_info_dic:
            search_info_dic[each[1]]=[]
        search_info_dic[each[1]].append(int(each[0]))

    dif_dic={}
    most_common_dif_count=0
    most_common_dif=-1
    for seq in seq_search_dic:
        for search_pos in search_info_dic[seq]:
            for got_pos in seq_search_dic[seq]:
                dif=search_pos-got_pos
                if dif not in dif_dic:
                    dif_dic[dif]=[]
                dif_dic[dif].append((seq,got_pos))
                if len(dif_dic[dif])>most_common_dif_count:
                    most_common_dif_count=len(dif_dic[dif])
                    most_common_dif=dif
    #### “we guarantee the correctness of modified labels in the dataset by using only 5-mers with equivalent positional differences in labels and DRS data”
    #### however, your data might not have enough 5-mers in a sample to produce enough differences. Therefore, you can discard the filtering step,
    #### and select samples with the lowest difference within a bound instead. The label quality would decline if doing so, as well as model's effectiveness.
    if most_common_dif_count<2:
        return
    all_select_info=dif_dic[most_common_dif]
        
    recv_num_2=len(gene_readindex_rows)*len(all_select_info)
    parent_conn_2,child_conn_2=multiprocessing.Pipe()
    receiver_2=multiprocessing.Process(target=receive_partition,args=(recv_num_2,file,gene,all_select_info,parent_conn_2,locks))
    receiver_2.start()
    
    for _,index_info in gene_readindex_rows:
        with locks['read'],open(file+'.txt','r') as f:
            f.seek(index_info['file_pos_start'],0)
            read_str=f.read(index_info['file_pos_end']-index_info['file_pos_start'])
        read_info=pd.read_csv(StringIO(read_str),delimiter='\t',index_col=False,
                        names=['contig','position','reference_kmer','read_index','strand','event_index',
                               'event_level_mean','event_stdv','event_length','model_kmer','model_mean',
                               'model_stdv','standardized_level','start_idx','end_idx'])
        for select_info in all_select_info:
            task_queue_2.put((read_info,select_info,child_conn_2))
    receiver_2.join()

def parallel_process_positive(file,restrict_file,n_processes_0,n_processes_1,n_processes_2):
    lock_0=dict()
    for lock_type in ['read','write','pipe_1','pipe_2']:
        lock_0[lock_type]=multiprocessing.Lock()
    task_queue_0=multiprocessing.JoinableQueue(maxsize=n_processes_0*2)
    consumers_0=[parallels.Consumer(task_queue=task_queue_0,task_function=parallel_main,locks=lock_0) for i in range(n_processes_0)]
    for process_0 in consumers_0:
        process_0.start()
  
    open(file+'_processed.json','w')
    open(file+'_processed.index','w')

    manager=multiprocessing.Manager()
    task_queue_1=manager.JoinableQueue(maxsize=n_processes_1*2)
    consumers_1=[parallels.Consumer(task_queue=task_queue_1,task_function=parallel_seq_search,locks=lock_0) for i in range(n_processes_1)]
    for process_1 in consumers_1:
        process_1.start()

    task_queue_2=manager.JoinableQueue(maxsize=n_processes_2*2)
    consumers_2=[parallels.Consumer(task_queue=task_queue_2,task_function=parallel_get_partition,locks=lock_0) for i in range(n_processes_2)]
    for process_2 in consumers_2:
        process_2.start()
    
    restrict_gene_dic={}
    with open(restrict_file+'.txt','r') as f:
        for line in f.readlines():
            items=line.strip().split('_')
            if items[0] not in restrict_gene_dic:
                restrict_gene_dic[items[0]]=[]
            restrict_gene_dic[items[0]].append((items[1],items[2]))

    for key in list(restrict_gene_dic.keys()):
        if len(restrict_gene_dic[key])==1:
            del restrict_gene_dic[key]

    index_file=pd.read_csv(file+'.index')
    genes=list(dict.fromkeys(index_file['transcript_id'].values.tolist()))
    index_file=index_file.set_index('transcript_id')

    for gene in genes:
        if gene[:15] not in restrict_gene_dic:
            continue
        read_index_l_info=list(index_file.loc[[gene]].iterrows())
        if len(read_index_l_info)<50:
            continue
        print(gene,len(read_index_l_info))
        task_queue_0.put((file,gene[:15],read_index_l_info[:500],restrict_gene_dic[gene[:15]],task_queue_1,task_queue_2))

    for _ in range(n_processes_0):
        task_queue_0.put(None)

    task_queue_0.join()

    for _ in range(n_processes_1):
        task_queue_1.put(None)
    for _ in range(n_processes_2):
        task_queue_2.put(None)


if __name__ == '__main__':
    ####change the file names here
    for nano_file in ['../events/SGNex_Hct116_directRNA_replicate3_run4-ref.eventalign','../events/SGNex_Hct116_directRNA_replicate4_run3-ref.eventalign','../events/SGNex_Hct116_directRNA_replicate3_run1-ref.eventalign']:
        restrict_file='./m6Asites_ENST'
        parallel_process_positive(nano_file,restrict_file,N_processes_0,N_processes_1,N_processes_2)

