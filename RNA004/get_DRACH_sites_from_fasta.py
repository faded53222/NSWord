import argparse
from Bio import SeqIO

def get_sites(fasta_name,save_name):
	with open(save_name,'w') as f:	
		for record in SeqIO.parse(fasta_name,"fasta"):
			save_id=record.id.split('_')[-1]
			seq=record.seq
			for i in range(0,len(seq)-4):
				kmer=seq[i:i+5]
				if kmer[0] in ['A','G','T'] and kmer[1] in ['A','G'] and \
				 kmer[2]=='A' and kmer[3]=='C' and kmer[4] in ['A','C','T']:
					f.write(f'{save_id}_{kmer}_{i}\n')

if __name__ == '__main__':
	parser=argparse.ArgumentParser(description="Get DRACH sites from fasta file")
	parser.add_argument('-i','--input',required=True,help="Input fasta file path")
	parser.add_argument('-o','--output',required=True,help="Output sites file path")

	args=parser.parse_args()
	print('begin getting DRACH sites from',args.input.split('/')[-1])
	get_sites(args.input,args.output)
	print('done getting DRACH sites from',args.input.split('/')[-1])
