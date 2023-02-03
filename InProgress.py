from Bio.Blast.Applications import NcbiblastnCommandline
from Bio import SeqIO, Align, SeqRecord
from itertools import combinations, islice
from argparse import ArgumentParser
import queue, time
from multiprocessing import Queue, Process, current_process
import numpy as np
import sys, os

#-------------------#
### Define Parser ###
#-------------------#

parser = ArgumentParser(description="Virome structural similarity identifier")
parser.add_argument("-i", action="store", dest="query_file", type=str, help="Provide input file as fasta format")
parser.add_argument("-o", action="store", dest="output_file", type=str, help="Provide input file as fasta format")

args = parser.parse_args()
infile = args.query_file
outfile = args.output_file

#------------------#
### class virome ###
#------------------#

class Worker(Process):
    def __init__(self, name, queue, infile):
        super().__init__()
        self.name = name
        self.queue = queue
        self.pairs = combinations( enumerate(SeqIO.parse(infile, "fasta")), 2)

    def run(self):
        print(f"Started {self.name}")
        count = 0
        while True:
            try:
                pair = self.queue.get(timeout=5)
                self.pairwise(pair)
                count += 1
            except Empty:
                break
        print(f"{self.name} done with {count} records")
    
    # static method
    def pairwise(self, pair):
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        align = aligner.align(pair[0][1].seq, pair[1][1].seq)
        return align.score
    
    def load_record_into_queue(self, queue):
        for pair in self.pairs:
            queue.put(pair)

def pairwise(input, output):
    for pair in iter(input.get, 'STOP'):
        results = []
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        align = aligner.align(pair[0][1].seq, pair[1][1].seq)
        results.append(align.score)
    output.put([score for score in results])

# comb tuple
#pairs = combinations( enumerate(SeqIO.parse(infile, "fasta")), 2)

"""
## parallel ##
def parallel():
    number_of_task = SIZE
    number_of_processes = 4
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []

    for pair in pairs:
        tasks_to_accomplish.put(pair)
    
    # creating processes
    for w in range(number_of_processes):
        Process(target=pairwise, args=(tasks_to_accomplish, tasks_that_are_done)).start()

    # print the output
    for i in range(number_of_processes):
        print(tasks_that_are_done.get())

    # stopping children
    for i in range(number_of_processes):
        tasks_to_accomplish.put('STOP')
"""
