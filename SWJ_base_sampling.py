#!/usr/bin/env python

import random
import math
import numpy
import itertools
import operator



def get_truth(inp, relate, cut):
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '=': operator.eq}
    return ops[relate](inp, cut)

class Table:
    def __init__(self, rows):
        # each row is an array of integers, and rows is an array of
        # rows, all of the same length
        self.rows = rows
        # we'll be inefficient and build an index for every column
        self.indices = []
        for c in self.rows[0]:
            self.indices.append({})
        for i, row in enumerate(self.rows):
            for column, value in enumerate(row):
                self.indices[column].setdefault(value, []).append(i)

def update_weights(ws,wc, path, path_ps, last_column_value,tables):
	#if 0, consider failed, prune path
	if last_column_value == 0:
		i = len(path)-1
		ws[i][path[i]] = 0
		row = tables[i].rows[path[i]]	#chosen row we failed at
		join_column_value = row[0]	#first column of that row, we joined on this with last col from left table
		#Search back through path.  If only one index out, push 0 weight to this table
		for i in xrange(len(path)-2, -1, -1):
		    row = tables[i].rows[path[i]]
		    matchingRows = tables[i].indices[-1].get(join_column_value,[])
		    count=0
		    for m in matchingRows:
			if ws[i+1][m] == 0:
			    count+=1
			else:
   			    break

		    if count == len(matchingRows):	#all other paths from this node has failed.  push 0 weight up
    		        ws[i][path[i]] = 0 
			join_column_value = row[0]
		    else:
			break

	else:
		total_p = 1.0
		for i in xrange(len(path)-1, -1, -1):
		    row = path[i]
		    wc[i][row] += 1.0
		    ws[i][row] += last_column_value / total_p
		    total_p *= path_ps[i]

##############################################################################
# this assumes that along the list of tables, we're joining the last
# column of the n-th table with the first column of the n+1-th table
#
# This is basic wanderjoin
def sample_path(tables, failCount):
    # start with sampling paths uniformly from the first column of
    # first table
    row = random.choice(tables[0].rows)
    last_column_value = row[-1]
#for first table, if part >7, return 0
    if row[0] > 7:
	failCount[0]+=1
	return 0
    p = 1.0 / len(tables[0].rows)
    for i, right_table in enumerate(tables[1:]):
        matching_rows = right_table.indices[0].get(last_column_value, [])
        if len(matching_rows) == 0:
            return 0
        p *= 1.0 / len(matching_rows)
        row_index = random.choice(matching_rows)
        last_column_value = right_table.rows[row_index][-1]
	if last_column_value >10:
	    failCount[1]+=1
	    return 0
    return last_column_value / p

################################################################################
# this is nonuniform wander-join

def sample_path_weighted(tables, weights, verbose=False):
    # start with sampling paths from the first column of
    # first table
    sub_weights = weights[0]
    sws = float(sum(sub_weights))
    sub_weights = list(w / sws for w in sub_weights)
    if verbose:
        print "sub_weights for table 0 selection:"
        print "  %s" % sub_weights
    row_index = numpy.random.choice(len(tables[0].rows), p=sub_weights)
    row = tables[0].rows[row_index]
    p = sub_weights[row_index]
    last_column_value = row[-1]

    if verbose:
        print "Sampled %s (w = %s)" % (row, p)
    for i, right_table in enumerate(tables[1:]):
        matching_rows = right_table.indices[0].get(last_column_value, [])
        if len(matching_rows) == 0:
            last_column_value=0
	else:
            sub_weights = list(weights[i+1][m] for m in matching_rows)
            sws = float(sum(sub_weights))
            sub_weights = list(w / sws for w in sub_weights)
            if verbose:
                print "matching rows for table %d" % (i+1)
		print "  %s" % matching_rows
		print "sub_weights for table %d selection:" % (i+1)
		print "  %s" % sub_weights
	    next_row_index = numpy.random.choice(len(matching_rows), p=sub_weights)
	    next_p = sub_weights[next_row_index]
	    last_column_value = right_table.rows[matching_rows[next_row_index]][-1]
	    p *= next_p
            if verbose:
		print "Sampled %s (w = %s, total %s)" % (right_table.rows[next_row_index], next_p, p)
    if verbose:
        print "Will return %s" % (last_column_value / p)
        print
    return last_column_value / p



def sample_path_dynamic_weights_sum(failed_path,tables, whereClauses,(weight_sums, weight_counts), verbose=False):
    # start with sampling paths from the first column of
    # first table
    sub_weights = list(s / c for (s, c) in itertools.izip(weight_sums[0], weight_counts[0]))
    sws = float(sum(sub_weights))
    sub_weights = list(w / sws for w in sub_weights)
    if verbose:
        print "sub_weights for table 0 selection:"
        print "  %s" % sub_weights
    path = []
    path_ps = []
    row_index = numpy.random.choice(len(tables[0].rows), p=sub_weights)
    path.append(row_index)
    row = tables[0].rows[row_index]
    p = sub_weights[row_index]
    path_ps.append(p)
    last_column_value = row[-1]
    current_table_index = 0

    if verbose:
        print "Sampled %s (w = %s)" % (row, p)

    #check where clause at each table
    whereOp = whereClauses[0]
    if len(whereOp) > 0:	#there is a where clause for this first table
	if get_truth(row[whereOp[0]],whereOp[1],whereOp[2])==False:	#if this row does not match the predicate, set to 0
	    last_column_value =0
            failed_path[0][row_index] +=1
        else:
            for i, right_table in enumerate(tables[1:]):
                matching_rows = right_table.indices[0].get(last_column_value, [])
    
     	        if len(matching_rows) == 0:	#we failed!
                    last_column_value = 0
		    failed_path[0][row_index] +=1
                else:
		   	sub_weights = list((weight_sums[i+1][m] / weight_counts[i+1][m]) for m in matching_rows)
			sws = float(sum(sub_weights))
			sub_weights = list(w / sws for w in sub_weights)
			if verbose:
			    print "matching rows for table %d" % (i+1)
			    print "  %s" % matching_rows
			    print "sub_weights for table %d selection:" % (i+1)
			    print "  %s" % sub_weights
			next_row_index = numpy.random.choice(len(matching_rows), p=sub_weights)
			path.append(matching_rows[next_row_index])
			next_p = sub_weights[next_row_index]
			path_ps.append(next_p)
			row = right_table.rows[matching_rows[next_row_index]];
			last_column_value = right_table.rows[matching_rows[next_row_index]][-1]
			#print " table %s row %s " % (i,right_table.rows[matching_rows[next_row_index]])

			#check whereClause
			whereOp = whereClauses[i+1]
			if len(whereOp) > 0:
			    if get_truth(row[whereOp[0]],whereOp[1],whereOp[2])==False:	#if this row does not match the predicate, set to 0
 	    		 	last_column_value =0;
				failed_path[i+1][matching_rows[next_row_index]] +=1


	#for testing with 4 row dataset
	#		if i==1:
	#		    if right_table.rows[matching_rows[next_row_index]][-1] == 4:
	#		         last_column_value=0
		
			p *= next_p
			if verbose:
			    print "Sampled %s (w = %s, total %s)" % (right_table.rows[matching_rows[next_row_index]], next_p, p)
    if verbose:
        print "Will return %s" % (last_column_value / p)
        print path_ps
        print path
        print

    update_weights(weight_sums,weight_counts,path, path_ps,last_column_value,tables)

    return last_column_value / p


##############################################################################


def swj_sum(tables, whereClauses, n=100000,verbose=False):
    # initialize dynamic weights uniformly

    weight_sums = []
    weight_counts = []
    failed_path=[]
    index = 0
    count = {}
    
    for t in tables:        
        weight_sums.append([1.0] * len(t.rows))
        weight_counts.append([1.0] * len(t.rows))
        failed_path.append([0.0] * len(t.rows))
	index+=1
#    weight_sums.append([43.0] * len(tables[0].rows))
#    weight_sums.append([2.6] * len(tables[1].rows))

    sx = 0
    sxx = 0
    ex=0
    exx=0
    for i in xrange(n):
        p = sample_path_dynamic_weights_sum(failed_path,tables, whereClauses,(weight_sums, weight_counts),verbose)
        sx += p
        sxx += p * p
	ex = sx / (float(i)+1)
	exx = sxx / (float(i)+1)
	print "%s: %s,%s,%s" % ((float(i)+1),ex,(exx-(ex*ex)),(1.96 * ((exx-(ex*ex)) ** 0.5) / ((i+1) ** 0.5))/(ex+1)  )

    # return 95% CI from large-sample confidence interval
    v = exx - ex * ex

    hw = 1.96 * (v ** 0.5) / (n ** 0.5)
    relError = hw/ex

    for (s, c) in itertools.izip(weight_sums, weight_counts):
        print " ",
        for (sv, cv) in itertools.izip(s, c):
	    
            print ("%.2f" % (sv / cv)),
	print

    table0sum = sum(weight_sums[0])
    table1sum = sum(weight_sums[1])
    table0count = sum(weight_counts[0])
    table1count = sum(weight_counts[1])

    print "sums: %s %s" %(table0sum/float(table0count), table1sum/float(table1count))
    countFailed = 0
    countFailedMult = 0
    totalFailures = 0
    for i in xrange(0,len(failed_path[0]),1):
	totalFailures+=failed_path[0][i]
        if failed_path[0][i] >0:
	    countFailed +=1
	if failed_path[0][i] >1:
	    countFailedMult +=1
    print "table 0, Total failed %f : %f Failed: %f multiple: %f" % (n,totalFailures,countFailed,countFailedMult)

    countFailed = 0
    countFailedMult = 0
    totalFailures = 0
    for i in xrange(0,len(failed_path[1]),1):
	totalFailures+=failed_path[1][i]
        if failed_path[1][i] >0:
	    countFailed +=1
	if failed_path[1][i] >1:
	    countFailedMult +=1
    print "table 1, Total failed %f : %f Failed: %f multiple: %f" % (n,totalFailures,countFailed,countFailedMult)

    return ex - hw, ex + hw, v, relError


def wander_join(tables, n=100000):
    sx = 0
    sxx = 0
    failCount = {}
    failCount[0]=0
    failCount[1]=0
    for i in xrange(n):
        p = sample_path(tables, failCount)
        sx += p
        sxx += p * p
    ex = sx / float(n)
    exx = sxx / float(n)

    # return 95% CI from large-sample confidence interval
    v = exx - ex * ex

    hw = 1.96 * (v ** 0.5) / (n ** 0.5)
    relError = hw/ex
    print "fail: %s" %failCount
    return ex - hw, ex + hw, v, relError

def join(tables):
    if len(tables) == 1:
        return tables[0]
    left = tables[0]
    right = tables[1]
    new_rows = []
    for row_left in left.rows:
        for row_right in right.rows:
            if row_left[-1] == row_right[0]:
                new_rows.append(row_left + row_right[1:])
    return join([Table(new_rows)] + tables[2:])


if __name__ == '__main__':
    whereClauses = {}
    whereClauses[0] =[]
    whereClauses[1] = []
    whereClauses[2] = []

    #print swj_sum(s2,whereClauses,10,True)
    numLines=0;
    nation = []
    f = open( './data/tpch/nation.csv', 'rU' ) #open the file in read universal mode
    for line in f:
        cells = line.split( "|" )
	numLines+=1
	#key, name, regionkey
        nation.append( [cells[ 0 ], cells[ 1 ], cells[ 2 ]]  ) 
    f.close()
    print "nation #rows %s",numLines
   
    numLines=0
    count=0
    part = []
    f = open( './data/tpch/part.csv', 'rU' ) #open the file in read universal mode
    for line in f:
        cells = line.split( "|" )
	numLines+=1
	#key, mftr, brand,size
       # part.append(  [cells[ 0 ], cells[ 2 ], cells[ 3 ], cells[5]] )
        part.append(  [float(cells[5]),float(cells[0])] )
	if float(cells[5]) <=7:
	    count +=1
    f.close()
    print "part #rows %s",numLines
    print "part <= 7 %s",count

    numLines=0
    ordertotal=0
    orders = []
    f = open( './data/tpch/orders.csv', 'rU' ) #open the file in read universal mode
    for line in f:
        cells = line.split( "|" )
	numLines+=1
	ordertotal += float(cells[3])
	#key, customerkey,orerstatus,totalprice
        #orders.append( [ cells[ 0 ], cells[ 1 ], cells[ 2 ], cells[3] ] )
        orders.append( [ float(cells[ 0 ]), float(cells[3]) ] )
    f.close()
    print "orders #rows %s" % numLines    
    print "order price total %s" % ordertotal

    numLines=0
    count=0
    totalQty = 0
    lineitem = []
    lineitem2 = []
    f = open( './data/tpch/lineitem.csv', 'rU' ) #open the file in read universal mode
    for line in f:
        cells = line.split( "|" )
	numLines+=1
	#orderkey, partkey,qty,price
        #lineitem.append( [ cells[ 0 ], cells[ 1 ], cells[ 4 ], cells[5] ] )
	lineitem.append( [ float(cells[ 1 ]), float(cells[ 4 ]), float(cells[0]) ] )
	lineitem2.append( [ float(cells[ 1 ]), float(cells[ 4 ]) ] )
	if (float(cells[ 4 ]) <=10):
  	    count +=1  
	totalQty +=float(cells[ 4 ])
    f.close()
    print "lineItem #rows %s",numLines
    print "lineItem #<10 sold %s",count

#repeat the column to join on/sum in last column
#query: Sum order totals for part size >7 and lineqty > 10 (total order$$ for parts with size>7 and more than 10 were bought at once

    tbl1 = [Table(part),
        Table(lineitem),
	Table(orders)]
#count num of parts sold
    tbl2 = [Table(part),
        Table(lineitem2)]    

    whereClauses[0] = [0,'<=',7]
    whereClauses[1] = [1,'<=',10]
    print whereClauses

    print "swj sum (without precomputation)"
    print swj_sum(tbl2,whereClauses,10000)

    print "regular wander join"
    print wander_join(tbl2,10000)
 

