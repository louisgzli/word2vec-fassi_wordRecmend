# coding: utf8
# Copyright (C) 2020 OPPO Inc. All rights reserved.
#
# Author: Ligengze (ligengze@faiss_index.com)
''' 生成faiss扩展服务需要的相关文件。
 输入为关键词向量文件，每行一个，第1列为关键词，后面为DIMENSION个浮点数
 设数据源名称为data_src，则输出三个文件
   1. $data_src.kw2id.txt  每行为<编号,关键词>
   2. $data_src.vec.dat    存有K*D个浮点数，文件大小为4*K*D
   1. $data_src.ivf        faiss的索引文件
'''


import codecs
import logging
import multiprocessing
import os
from array import array
import json
import numpy as np
import faiss

logging.basicConfig(format='[%(asctime)s] %(levelname)s line %(lineno)d - %(message)s')
logger = logging.getLogger(__name__)


#keyword embedding 里关键词的的维度
DIMENSION = 300


def prepare_kw_id_vec_(ifn_vec_txt, ofn_vec_bin, ofn_kw_id):
    ''' 从<关键词，向量>文本文件中生成<关键词,id>及二进制的向量文件
    Args:
      ifn_vec_txt: <关键词，向量>文本文件
      ofn_vec_bin: 二进制的向量文件
      ofn_kw_id: <关键词,id>文件
    Returns:
      <关键词,id> map
    '''
    if os.path.exists(ofn_vec_bin):
        os.remove(ofn_vec_bin)
    ofh_vec_bin = open(ofn_vec_bin, 'wb')
    ifh_vec_text = codecs.open(ifn_vec_txt, 'r', 'utf8')
    fs_header = ifh_vec_text.readline().split()
    _, dimension = int(fs_header[0]), int(fs_header[1])
    if dimension != DIMENSION:
        logger.fatal('File %s: expect dimension %d got %d',
                     ifn_vec_txt, DIMENSION, dimension)
    kw2id = {}
    kw_list = []
    for iline, line in enumerate(ifh_vec_text, 2):
        # if len(kw2id) > 1000:
        #     break
        if iline % 10000 == 0:
            logger.info('%s: %d ', ifn_vec_txt, iline)
        fs = line.strip().split(' ')
        if len(fs) != 1 + DIMENSION:
            logger.error('%s:%d: expect %d fields, got %d',
                         ifn_vec_txt, iline, 1 + DIMENSION, len(fs))
            continue
        if fs[0] in kw2id:
            logger.warn('%s:%d: Dup kw [%s] with line %d#',
                        ifn_vec_txt, iline, fs[0], kw2id[fs[0]])
            fs[0] += "#%d" % iline  # 确保不重复
        kw2id[fs[0]] = len(kw2id)
        kw_list.append(fs[0])
        array('f', [float(f) for f in fs[1:]]).tofile(ofh_vec_bin)
    logger.info("writing to %s", ofn_kw_id)
    ofh_kw_id = codecs.open(ofn_kw_id, 'w', 'utf8')
    for kid, kw in enumerate(kw_list, 0):
        print (ofh_kw_id, '%d\t%s' % (kid, kw))
    return kw2id


def prepare_index_(ifn_vec_bin, n_kw, index_param, ofn_index):
    ''' 从二进制向量文件构建faiss索引
    Args:
      ifn_vec_bin 二进制向量文件；注意由于使用L2距离，向量要求已归一化
      n_kw 向量个数
      index_param 构建索引的参数串，见faiss相关说明
      ofn_index 索引文件
    '''
    n_float = n_kw * DIMENSION
    logger.info('loading %d float from %s', n_float, ifn_vec_bin)
    vecs = array('f')
    vecs.fromfile(open(ifn_vec_bin, 'rb'), n_float)
    if len(vecs) != n_float:
        logger.error('File %s expect %d float, got %d',
                     ifn_vec_bin, n_float, len(vecs))
        return

    # 一维转成n_kw * DIMENSION的二维
    data = np.frombuffer(vecs, dtype=np.float32).reshape((-1, DIMENSION))
    assert data.shape[0] == n_kw
    index = faiss.index_factory(DIMENSION, index_param, faiss.METRIC_L2)
    index.verbose = True
    index.do_polysemous_training = False  # PQ only

    # 训练及建索引
    index.train(data)
    index.add(data)

    logger.info('saving index into %s', ofn_index)
    faiss.write_index(index, ofn_index)

    logger.info('done %s', ofn_index)


def do_build(ifn_vec_txt, index_param, ofn_kw_id, ofn_index):
    logger.info('preparing ' + ifn_vec_txt)
    fn_vec_bin = ifn_vec_txt + '.bin'
    kw2id = prepare_kw_id_vec_(ifn_vec_txt, fn_vec_bin, ofn_kw_id)
    jsObj = json.dumps(kw2id)
    fileObject = open('./jsonFile.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()
    prepare_index_(fn_vec_bin, len(kw2id), index_param, ofn_index)
    logger.info('done %s', ifn_vec_txt)
    logger.info('You may want to delete tmp file %s', fn_vec_bin)





def build_louisgzli():
    '''
    # louisgzli 使用all.keyword.64.emb.20180827构建离线ann索引
    :return:
    '''

    threads_list = []

    index_param = 'IVF100,PQ10'
    fn_vec_txt = "./word_embedding.txt"
    # 输出1：keyword到id
    fn_kw_id = './kw2id.txt'
    # 输出3：faiss的索引文件
    fn_index = './index.ivf'
    args = (fn_vec_txt, index_param, fn_kw_id, fn_index)
    thread = multiprocessing.Process(target=do_build, args=args)
    thread.start()
    threads_list.append(thread)
    for thread in threads_list:
        thread.join()

if __name__ == '__main__':
    import time
    start = time.time()
    build_louisgzli()
    print("total time consumption: {}".format(time.time()-start))