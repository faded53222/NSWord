import argparse
import random
from pyensembl import EnsemblRelease
import time

def ENSG_reindex_to_ENST(Ensembl_data,sites_file):
    sites_write_ENST=open(sites_file+'_ENST.txt','w')
    f=open(sites_file+'.txt','r')

    c_gene=None
    c_trans={}
    while 1:
        line=f.readline()
        if not line:
            break
        gene,g_pos,k5=line.strip().split('_')
        g_pos=int(g_pos)

        #time.sleep(0.1)
        try:
            Ensembl_data.transcript_ids_of_gene_id(gene)
        except:
            continue

        if gene!=c_gene:
            c_trans={}
            for t_trans in Ensembl_data.transcript_ids_of_gene_id(gene):
                c_trans[t_trans]=[]
                for t_exon in Ensembl_data.exon_ids_of_transcript_id(t_trans):
                    t_exon_info=Ensembl_data.exon_by_id(t_exon)
                    c_trans[t_trans].append([t_exon_info.start,t_exon_info.end])
            c_gene=gene

        for ENST in c_trans:
            c_pos=0
            for start_end in c_trans[ENST]:
                if g_pos>start_end[1]:
                    c_pos+=start_end[1]-start_end[0]+1
                    continue
                if g_pos>=start_end[0]:
                    sites_write_ENST.write('%s_%d_%s\n'%(ENST,c_pos+g_pos-start_end[0],k5))                
                break
    sites_write_ENST.close()
    f.close()

if __name__ == '__main__':
	parser=argparse.ArgumentParser(description="ENSG sites to ENST sites")
	parser.add_argument('-i','--input',required=True,help="Input ENSG file")
	args=parser.parse_args()

	sites_file=args.input
	Ensembl_data=EnsemblRelease(110)
	print('begin ENSG to ENST')
	ENSG_reindex_to_ENST(Ensembl_data,sites_file)
	print('done ENSG to ENST')
