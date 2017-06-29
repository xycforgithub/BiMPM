##ensemble restore
"""
Restore answer spans into the submit format
"""
import argparse
import codecs
import json
import io
import sys

def ReadFile(path, encoding='utf-8'):
    with codecs.open(path, 'r', encoding=encoding) as reader:
        return reader.readlines()

def WriteFile(path, text, encoding='utf-8'):
    with codecs.open(path, 'w', encoding=encoding) as writer:
        writer.write(text)

def Parse2DSpans(text):
    span_list = []
    for line in text:
        line = line.strip()
        sents = line.split('#')
        sub_span_list = []
        for sent in sents:
            _s = []
            tokens = sent.split()
            for token in tokens:
                items = token.split(':')
                _s.append(( int(items[0]), int(items[1]) ))
            sub_span_list.append(_s)
        span_list.append(sub_span_list)
    return span_list


def ParsePos(text):
    _list = []
    for line in text:
        pos = [int(t) for t in line.split('\t')[:2]]
        _list.append(pos)
    return _list

def SearchPos(span_list, start, stop):
    spans = []
    cnt = 0
    for sent in span_list:
        for token in sent:
            if start == cnt:
                spans.append(token)
            if stop == cnt:
                spans.append(token)
            cnt += 1
    #print len(spans),start,stop;
    #print spans, start, stop, cnt
    assert len(spans) == 2

    return (spans[0][0], spans[1][1])

def get_best_span_fast(ypi, yp2i):
    max_val = 0
    argmax_start = 0;
    best_word_span = (0, 1)
    for idx, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):
            val1 = ypi[argmax_start]
            if val1 < ypif:
                val1 = ypif
                argmax_start = idx
            val2 = yp2if
            if val1 * val2 > max_val:
                best_word_span = (argmax_start, idx)
                max_val = val1 * val2
    return (best_word_span[0], best_word_span[1], float(max_val))
    
def get_best_span(ypi, yp2i, maxSpan):
    if(maxSpan == 0):
        return get_best_span_fast(ypi, yp2i)
    max_val = 0
    argmax_start = 0
    argmax_end = 1
    for idx_i in range(0, len(ypi)):
        for idx_j in range(idx_i, len(yp2i)):
            if(idx_j - idx_i >= maxSpan):
                continue
            
            val1 = ypi[idx_i]
            val2 = yp2i[idx_j]
            
            if val1 * val2 > max_val:
                argmax_start = idx_i
                argmax_end = idx_j
                max_val = val1 * val2
    return (argmax_start, argmax_end, float(max_val))

    
def parseProb(items, sp, ep): 
    for item in items:
        token = item.strip().split('#')
        tokenId = int(token[0])
        startP = float(token[1])
        endP = float(token[2])
        sp[tokenId] += startP
        ep[tokenId] += endP


def Ensemble(num, fileList, spans, raw, ids_list, outputFile, maxspan):
    files = [];
    for i in range(0, num):
        files.append(codecs.open(fileList[i], 'r', 'utf-8'))  # file(fileList[i]))
    
    results = {}
    
    idx = 0
    for line1 in files[0]:
        #print idx
        span = spans[idx]
        cont = raw[idx]
        ids = ids_list[idx].strip()
        
        items1 = line1.strip().split(' ')
        sp = [0] * len(items1)
        ep = [0] * len(items1)
        parseProb(items1, sp, ep)
        sidx, eidx, score = get_best_span(sp, ep, maxspan)
        
        start, stop = SearchPos(span, sidx, eidx)
        assert stop >= start
        phrase = cont[start:stop]
        
        d = {}
        d[phrase] = score
        maxScore = score
        maxphrase = phrase
        for i in range(1, num):
            linei = files[i].readline()
            itemsi = linei.strip().split(' ')
            spi = [0] * len(itemsi)
            epi = [0] * len(itemsi)
            parseProb(itemsi, spi, epi)
            sidx, eidx, score = get_best_span(spi, epi, maxspan)
            
            start, stop = SearchPos(span, sidx, eidx)
            assert stop >= start
            phrase = cont[start:stop]
            
            if(phrase not in d):
                d[phrase] = 0
            d[phrase] += score
            if(d[phrase] > maxScore):
                maxScore = d[phrase]
                maxphrase = phrase 
        results[ids] = maxphrase
        idx = idx + 1
    with codecs.open(outputFile, 'w', 'utf-8') as writer:
        json.dump(results, writer)

def Restore(span_path, raw_path, id_path, fin_path, fout_path):
    spans = Parse2DSpans(ReadFile(span_path))
    raw = ReadFile(raw_path)
    ids_list = ReadFile(id_path)
    pos_data = ParsePos(ReadFile(fin_path))
    assert len(spans) == len(raw) and len(raw) == len(ids_list)
    assert len(ids_list) == len(pos_data)
    results = {}
    for idx, pos in enumerate(pos_data):
        ids = ids_list[idx].strip()
        cont = raw[idx]
        span = spans[idx]
        start, stop = SearchPos(span, pos[0], pos[1])
        assert stop >= start
        ans = cont[start:stop]
        results[ids] = ans
    with codecs.open(fout_path, 'w', 'utf-8') as writer:
        json.dump(results, writer)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reform to submit format')
    parser.add_argument('--span_path', required=True, help='The path of context spans')
    parser.add_argument('--raw_path', required=True, help='The path of raw context')
    parser.add_argument('--ids_path', required=True, help='The path of ids')
    parser.add_argument('--inpaths', required=True, help='ensemble input paths')
    #parser.add_argument('--ensembleStyle', required=True, help='ensemble style; 0, 1, 2')
    parser.add_argument('--fout', required=True, help='output file path')
    parser.add_argument('--seqlen', required=True, help='seqlen')
    args = parser.parse_args()
    
    spans = Parse2DSpans(ReadFile(args.span_path))
    raw = ReadFile(args.raw_path)
    ids_list = ReadFile(args.ids_path)
    pathList = args.inpaths.strip(',').split(',')
    
    nouse = 1;
    for i in range(0, len(pathList)):
        for j in range(0, len(pathList)):
            if( i == j):
                continue
            elif(pathList[i] == pathList[j]):
                nouse = 1;
            
    # if(nouse == 0):
    #     print("path number" + str(len(pathList)))
    #     print("seq length" + str(int(args.seqlen)))
    #     Ensemble(len(pathList), pathList, spans, raw, ids_list, args.fout, int(args.seqlen))
    
    
    Restore(args.span_path, args.raw_path, args.ids_path, args.inpaths, args.fout)
