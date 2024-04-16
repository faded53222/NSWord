####change the file names to those you want to process in main.
import multiprocessing
import pandas as pd
import numpy as np

import parallels

Chunk_Size=1000000
N_processes=10

def make_index(df,file_pos_start,out_path,lock):
    df=df.set_index(['contig','read_index'])
    _indexs=list(dict.fromkeys(df.index))
    df=df.reset_index()
    df=df.set_index(['read_index'])
    df=df.sort_index()
    
    file_pos_end=file_pos_start
    with lock,open(out_path,'a') as fw:
        for _index in _indexs:
            file_pos_end+=df.loc[_index[1]]['line_length'].sum()
            fw.write('%s,%s,%d,%d\n'%(_index[0],str(_index[1]),file_pos_start,file_pos_end))
            file_pos_start=file_pos_end

def parallel_make_index(file,chunk_size,n_processes):
    index_path=file+'.index'
    with open(index_path,'w') as fw:
        fw.write('transcript_id,read_index,file_pos_start,file_pos_end\n')

    lock=multiprocessing.Lock()
    task_queue=multiprocessing.JoinableQueue(maxsize=n_processes*2)
    consumers=[parallels.Consumer(task_queue=task_queue,task_function=make_index,locks=lock) for i in range(n_processes)]
    for process in consumers:
        process.start()

    event_file=open(file+'.txt','r')
    file_pos_start=len(event_file.readline())

    chunk_split=None
    index_features=['contig','read_index','line_length']
    for chunk in pd.read_csv(file+'.txt',chunksize=chunk_size,sep='\t'):
        chunk_complete=chunk[chunk['contig']!=chunk.iloc[-1]['contig']]
        chunk_concat=pd.concat([chunk_split,chunk_complete])
        chunk_concat_size=len(chunk_concat.index)
        lines=[len(event_file.readline()) for i in range(chunk_concat_size)]
        chunk_concat.loc[:, 'line_length']=np.array(lines)
        task_queue.put((chunk_concat[index_features],file_pos_start,index_path))
        file_pos_start+=sum(lines)
        chunk_split=chunk[chunk['contig']==chunk.iloc[-1]['contig']].copy()

    chunk_split_size=len(chunk_split.index)
    last_split_lines=[len(event_file.readline()) for i in range(chunk_split_size)]
    chunk_split.loc[:,'line_length']=np.array(last_split_lines)
    task_queue.put((chunk_split[index_features],file_pos_start,index_path))

    task_queue=end_queue(task_queue,n_processes)
    task_queue.join()

if __name__ == '__main__':
    ####change the file names here
    for nano_file in ['../events/SGNex_Hct116_directRNA_replicate3_run4-ref.eventalign','../events/SGNex_Hct116_directRNA_replicate4_run3-ref.eventalign','../events/SGNex_Hct116_directRNA_replicate3_run1-ref.eventalign']:
		parallel_make_index(nano_file,Chunk_Size,N_processes)
