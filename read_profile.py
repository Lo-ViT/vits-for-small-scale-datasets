import sys, os
import cProfile, pstats
import argparse

def init_parser():
    parser = argparse.ArgumentParser(description='Vit small datasets quick training script')
    parser.add_argument('--inpath', default=f'data', type=str)
    parser.add_argument('--outpath', default=f'data', type=str)
    return parser

def analyze_dmp(myinfilepath='stats.dmp', myoutfilepath='stats.log'):
    out_stream = open(myoutfilepath, 'w')
    ps = pstats.Stats(myinfilepath, stream=out_stream)
    sortby = 'cumulative'

    ps.strip_dirs().sort_stats(sortby).print_stats(.3)  # plink around with this to get the results you need

def main(args):   
    analyze_dmp(myinfilepath=args.inpath, myoutfilepath=args.outpath)

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    main(args)