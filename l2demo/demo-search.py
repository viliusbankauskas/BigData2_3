#!/usr/bin/env python
"""
demo usage
"""
import sqlite3,pickle,os

print("Loading search pickle data");

with open("page-rank-calculated.pickle",'rb') as f:
  page_rank_calculated = pickle.load(f)

with open("lookup-articles.pickle",'rb') as f:
  lookup = pickle.load(f)



if os.path.exists('database.db'):
  print("Removing old database");
  os.unlink('database.db')

# it can be file
db = sqlite3.connect('database.db')

c = db.cursor()
c.execute("""
create table pg_table (
  title text not null,
  pr real
  );
""")


for idx,val in enumerate(page_rank_calculated):
  if idx in lookup['backward'] :
    c.execute("insert into pg_table values (?,?)",(lookup['backward'][idx],val))

c.execute("create index idx1 on pg_table(pr);");
c.execute("create index idx2 on pg_table(title);")


db.commit()

print("press CTRL-c or CTRL-d to exit");

while True:
  term = input("Enter search term>>> ");

  term = c.execute("select title, pr from pg_table where title like ? order by pr desc limit 20", ("%"+term+"%",))

  print("Results:");
  for i in term:
    print("%s (pr %r)" % (i[0],i[1]))