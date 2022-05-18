#!/usr/bin/env python
"""
This program generates lookup table
"""
from mpi4py import MPI
import sys, datetime, os,time
import lzma,glob

from lxml import etree

comm = MPI.COMM_WORLD

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()


def get_titles(f):
  all_titles = []
  with open(f,'rb') as compressed:
    with lzma.LZMAFile(compressed) as uncompressed:
      #print(uncompressed.read())
      root = etree.parse(uncompressed).getroot()
      for page in root.xpath("page"):
        all_titles.append(page.find("title").text)
  return all_titles


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

print("rank %d processing: %d files" % (rank,len(the_chunks)))


the_titles = []

for i in the_chunks:
  the_titles = the_titles + get_titles(i)
  

the_titles = comm.reduce(the_titles,MPI.SUM,root=0)

if rank == 0:
  #print("Got",len(the_titles),"titles")
  
  
  forward_table = { the_titles[i]:i for i in range(len(the_titles))}
  backward_table = { i :the_titles[i] for i in range(len(the_titles))}
  
  print("Saving lookup table to lookup-articles.pickle");
  import pickle
  
  with open("lookup-articles.pickle",'wb') as f:
    pickle.dump({'forward' : forward_table, 'backward' : backward_table},f)
