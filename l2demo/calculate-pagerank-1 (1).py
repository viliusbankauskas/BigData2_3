#!/usr/bin/env python
"""
this program generates PR sparse matrix
"""
from mpi4py import MPI
import sys, datetime, os,time
import lzma,glob, pickle

import scipy.sparse

from pprint import pprint
import re
import numpy as np

from lxml import etree

comm = MPI.COMM_WORLD

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()


def get_alllinks(f,forward_table):
  all_links = {}
  with open(f,'rb') as compressed:
    with lzma.LZMAFile(compressed) as uncompressed:
      #print(uncompressed.read())
      root = etree.parse(uncompressed).getroot()
      for page in root.xpath("page"):
        title = page.find('title').text
        last_rev = page[-1]
        text = last_rev.find('text').text
        if text is None: continue # empty page, maybe just redirect
        links = re.findall(r"\[\[[^\]]+\]\]",text)
        
        links_ids = {}
        for i in links:
          i = i[2:-2].split('|')[0]
          if i in forward_table:
            idnum= forward_table[i]
            
            if idnum not in links_ids:
              links_ids[idnum] = 0
            
            links_ids[idnum] += 1
        
        all_links[ forward_table[title] ]=  links_ids
        
  return all_links


def list_split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

if rank == 0:
  print("Begin calculating title vector")
  files = sorted(glob.glob("../wiki_data/ltwiki-latest-pages-meta-history-*.xml.xz"))[1:] # ltwiki-latest-pages-meta-history-00.xml.xz it is index
  print("Got %d items" % len(files))

  the_chunks = list(list_split(files,size)) # world split
else:
  the_chunks = []

the_chunks = comm.scatter(the_chunks,root=0)
with open("lookup-articles.pickle",'rb') as f:
  lookup_table = pickle.load(f)

print("rank %d processing: %d files" % (rank,len(the_chunks)))


# assemble sparse matrix
data = []

for xml_split_name in the_chunks:
  pages_links_graph = get_alllinks(xml_split_name,lookup_table['forward'])
  
  for page_col in pages_links_graph:
    links_data = pages_links_graph[page_col].copy()
    if page_col in links_data:
      del links_data[page_col]

    if len(links_data) == 0: continue
    for i in links_data:
      data.append((page_col,i))


data = comm.reduce(data,root=0)


if rank == 0:
  graph = {'f' : {}, 'b' : {} } # forward and backward graph

  graph_f = graph['f']
  graph_b = graph['b']

  with open("wiki-links.txt",'w') as f:
    for i in data:
      f.write("%d\t%d\n" % i)
      if i[0] not in graph_f:
        graph_f[i[0]]=[]
      if i[1] not in graph_b:
        graph_b[i[1]]= []
      graph_f[i[0]].append(i[1])
      graph_b[i[1]].append(i[0])

  with open("wiki-graph.pickle",'wb') as f:
    pickle.dump(graph,f)

