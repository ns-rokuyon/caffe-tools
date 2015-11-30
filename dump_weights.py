# coding: utf-8
# Print weights for caffemodel
import sys, argparse
import caffe

def parse_args():
    parser = argparse.ArgumentParser(description='Dump weights in caffemodel')
    parser.add_argument('--model', required=True, help='Modeldef file')
    parser.add_argument('--layer', required=True, help='Layer name')
    parser.add_argument('--v1', action='store_true', help='Use old version caffe')
    parser.add_argument('pretrained', help='Caffemodel file')

    return parser.parse_args()


def main():
    args = parse_args()
    if args.v1:
        net = caffe.Net(args.model, args.pretrained)
    else:
        net = caffe.Net(args.model, args.pretrained, caffe.TEST)

    print(net.params[args.layer][0].data)


if __name__ == '__main__':
    main()

