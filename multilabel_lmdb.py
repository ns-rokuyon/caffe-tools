# coding: utf-8
import os, sys, argparse, random
import numpy as np
import cv2
import caffe
import lmdb


def parse_args():
    parser = argparse.ArgumentParser(description='Create lmdbs for multilabel classification task')
    parser.add_argument('--name', default=None, help='LMDB name')
    parser.add_argument('--resize', default=None, help='Image size to resize "w,h"')
    parser.add_argument('--image_root', default='', help='Root path to images in listfile')
    parser.add_argument('--padding', action='store_true', help='Zero padding to image (for ground truth)')
    parser.add_argument('--verbose', action='store_true', help='Print logs to stdout')
    parser.add_argument('listfile', help='Path to listfile')

    return parser.parse_args()


def swap_channel(img):
    ''' Swap channel order for caffe '''
    if img.ndim != 3:
        raise 'Dim=%d is not supported' % img.ndim
    return img.transpose((2,0,1))


def readimg(path, resize=None, grayscale=False):
    ''' 
        Load image file with OpenCV
            resize: Tuple (width, height)
    '''
    try:
        if grayscale:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if resize is not None:
                img = cv2.resize(img, resize)
            img = np.array([img])   # Shape: (1,H,W)
        else:
            img = cv2.imread(path)
            if resize is not None:
                img = cv2.resize(img, resize)
            img = swap_channel(img) # Shape: (C,H,W)
    except AttributeError as e:
        print('[ERROR] %s: %s' % (path, str(e)))
        raise
    return img


def zero_padding(img, max_wh):
    c,h,w = img.shape
    pad_h = max_wh - h
    pad_w = max_wh - w
    return np.pad(img, pad_width=((0,0), (0,pad_h), (0,pad_w)),
            mode='constant', constant_values=0)


def image_parser(line, args):
    path = os.path.join(args.image_root, line.split(' ')[0])
    if args.verbose:
        print(path)

    if args.resize is None:
        resize = None
    else:
        resize = tuple(map(int, args.resize.split(',')))

    img = readimg(path, resize=resize)
    if args.zero_padding:
        return zero_padding(img, 500)
    return img


def groundtruth_image_parser(line, args):
    path = os.path.join(args.image_root, line.split(' ')[1])
    if args.verbose:
        print(path)

    if args.resize is None:
        resize = None
    else:
        resize = tuple(map(int, args.resize.split(',')))

    img = readimg(path, resize=resize, grayscale=True)
    if args.zero_padding:
        return zero_padding(img, 500)
    return img


def label_parser(line, args):
    labels = line.split(' ')[1].split(',')
    if args.verbose:
        print(labels)
    return np.array([[labels]])     # Shape: (C=1,H=1,W=labels)


def get_parser(dbtype):
    if not dbtype in ['image', 'label']:
        raise 'Invalid dbtype=%d' % dbtype
    if dbtype == 'image':
        return image_parser
    if dbtype == 'label':
        return label_parser


def write_lmdb(name, parser, fp, skips, args):
    db = lmdb.open(name, map_size=int(1e12))
    with db.begin(write=True) as txn:
        for i, line in enumerate(fp):
            if parser == label_parser and i in skips:
                print('Skip: %s' % line)
                continue

            line = line.rstrip()

            if args.verbose and i % 100 == 0:
                print('Progress: i=%d' % i)

            try:
                arr = parser(line, args)
            except:
                print('Skip: %s' % line)
                if not i in skips:
                    skips.append(i)
                continue

            datum = caffe.io.array_to_datum(arr)
            txn.put('{:0>10d}'.format(i), datum.SerializeToString())
    db.close()


def dbname(args):
    if args.name is None:
        name = args.listfile.split('/')[-1].split('.')[0]
    else:
        name = args.name
    return '%s_%s.lmdb' % (name, dbtype)


def main():
    args = parse_args()

    skips = []
    for dbtype in ['image', 'label']:
        name = dbname(args)
        parser = get_parser(dbtype)

        with open(args.listfile) as fp:
            write_lmdb(name, parser, fp, skips, args)
            if args.verbose:
                print('Done: dbtype=%s' % dbtype)


if __name__ == '__main__':
    main()
