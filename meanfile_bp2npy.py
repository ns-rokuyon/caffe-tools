# coding: utf-8
# Convert mean file (.binaryproto -> .npy)
import os, sys, argparse
import numpy as np
import caffe

def parse_args():
    parser = argparse.ArgumentParser(description='Convert mean file')
    parser.add_argument('--save', default=None, help='Save file name')
    parser.add_argument('meanfile', help='Meanfile binaryproto')

    return parser.parse_args()


def load_binaryproto(bp):
    blob = caffe.proto.caffe_pb2.BlobProto()
    with open(bp, 'rb') as fp:
        blob.ParseFromString(fp.read())
    return np.squeeze(caffe.io.blobproto_to_array(blob))


def main():
    args = parse_args()

    bp = args.meanfile
    if args.save:
        npy = args.save
    else:
        npy = (os.path.basename(bp)).split('.')[0]

    np.save(npy, load_binaryproto(bp))
    print('Save to %s' % npy)


if __name__ == '__main__':
    main()
