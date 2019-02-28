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


#This is basic wanderjoin
def sample_path(tables, failCount):
    # start with sampling paths uniformly from the first column of
    # first table
    row = random.choice(tables[0].rows)
    last_column_value = row[-1]
    p = 1.0 / len(tables[0].rows)
    for i, right_table in enumerate(tables[1:]):
        matching_rows = right_table.indices[0].get(last_column_value, [])
        if len(matching_rows) == 0:
            return 0
        p *= 1.0 / len(matching_rows)
        row_index = random.choice(matching_rows)
        last_column_value = right_table.rows[row_index][-1]
    return last_column_value / p

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

############## group by ################

def prune_weights_group_by(ws, path, path_ps, last_column_value,tables,groupByCounts):
	#if 0, consider failed, prune path
	if last_column_value == 0:
		i = len(path)-1
		ws[i][path[i]] = 0
		row = tables[i].rows[path[i]]	#chosen row we failed at
		join_column_value = row[0]	#first column of that row, we joined on this with last col from left table
		#########need to check if failure was at first table, update groupcount accordingly)		
		if i == 0:
		    groupByVal = row[0]
		    groupByCounts[groupByVal] -=1
 		    # need to update all other rows with this groupBy value
		    matching_rows = tables[0].indices[0].get(groupByVal, [])
		    for m in matching_rows:
			ws[0][m] = min(ws[0][m],groupByCounts[groupByVal]) #don't overwrite if already 0, otherwise, reduce by 1		

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
			#if we're at a groupBy row, update groupBy counts
			# make this generic at some point
			if i == 0:
			    groupByVal = row[0]
			    groupByCounts[groupByVal] -=1
 			    # need to update all other rows with this groupBy value
			    matching_rows = tables[0].indices[0].get(groupByVal, [])
			    for m in matching_rows:
		   	        ws[0][m] = min(ws[0][m],groupByCounts[groupByVal]) #don't overwrite if already 0, otherwise, reduce by 1
			    
		    else: 
			break

  

def sample_path_dynamic_weights_sum_groupby(failed_path,sub_weights,updateFunction, whereClauses,weight_sums,
 groupByCounts,numGroupByKeys,verbose=False):
    # start with sampling paths from the first column of first table,#check for divide-by-0 (this will happen if a row gets pruned out)
    #sub_weights = list((0 if s==0 else (1/(s*float(numGroupByKeys)))) for s in weight_sums[0])
    #sub_weights=list( 1 for w in weight_sums[0]) #use this for uniform/regular WJ
    #sws = float(sum(sub_weights))
    #sub_weights = list( w/sws for w in sub_weights)
    if verbose:
        print "sub_weights for table 0 selection:"
        print "  %s" % sub_weights
    path = []
    path_ps = []
    row_index = numpy.random.choice(len(tbl2[0].rows), p=sub_weights)
    path.append(row_index)
    row = tbl2[0].rows[row_index]
    p = sub_weights[row_index]

    #For proper HT estimator calculation, use uniform probability based on how many in each category,  we are tricking the random.choice to pick rows at a different probability
    p = (1.0/groupByCounts[row[0]])

    path_ps.append(p)
    last_column_value = row[-1]
    current_table_index = 0

    #assume the groupBy condition is in the first column of first table
    groupByValue = row[0]

    if verbose:
        print "Sampled %s (w = %s)" % (row, p)

    #check where clause at each table
    whereOp = whereClauses[0]
    if len(whereOp) > 0 and get_truth(row[whereOp[0]],whereOp[1],whereOp[2])==False:	#if this row does not match the predicate, set to 0
	    last_column_value =0
            failed_path[0][row_index] +=1
    else:
            for i, right_table in enumerate(tbl2[1:]):
                matching_rows = right_table.indices[0].get(last_column_value, [])
    
     	        if len(matching_rows) == 0 or float(sum(list(weight_sums[i+1][m] for m in matching_rows))) == 0:	#we failed!
                    last_column_value = 0
		    failed_path[0][row_index] +=1
                else:
		   	sub_weights = list(weight_sums[i+1][m] for m in matching_rows)	
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
			#needs to be the uniform probability of picking a row, not the weighted
			next_p = 1.0/sum(weight_sums[i+1][m]>0 for m in matching_rows)
			path_ps.append(next_p)
			row = right_table.rows[matching_rows[next_row_index]];
			last_column_value = right_table.rows[matching_rows[next_row_index]][-1]
			#print " row %s row %s " % (row1,right_table.rows[matching_rows[next_row_index]])
			
			#check whereClause
			whereOp = whereClauses[i+1]
			if len(whereOp) > 0:
			    if get_truth(row[whereOp[0]],whereOp[1],whereOp[2])==False:	#if this row does not match the predicate, set to 0
 	    		 	last_column_value =0;
				failed_path[i+1][matching_rows[next_row_index]] +=1
	
			p *= next_p
			if verbose:
			    print "Sampled %s (w = %s, total %s)" % (right_table.rows[matching_rows[next_row_index]], next_p, p)
    if verbose:
        print "Will return %s" % (last_column_value / p)
        print path_ps
        print path
        print

    #updateFunction(weight_sums,path, path_ps,last_column_value,tables,groupByCounts)

    return last_column_value / p, groupByValue

def groupby_wander_join_sum(tables, whereClauses, updateFunction,initWeight,n=100000,verbose=False):
    weight_sums = []
    failed_path=[]
    countFailed = [0,0]
    countFailedMult = [0,0]
    totalFailures = [0,0]
    sx = {}
    sxx = {}
    ex={}
    exx={}
    v={}
    hw={}
    relError = {}
    numSamplesPerGroup = {}
    count={}
    for t in tables:
        weight_sums.append([1.0] * len(t.rows))
        failed_path.append([0.0] * len(t.rows))


    #find the groupby distribution
    groupByCounts = {}
    for group in tables[0].indices[0].keys():
	groupByCounts[group] = len(tables[0].indices[0].get(group))
        sx[group] = 0
    	sxx[group] = 0
	ex[group] = 0
	exx[group] =0
	count[group]=0
	numSamplesPerGroup[group]=0
	#update weights on matching rows
        # for now, the "weight" is the number of this type of groupby.
        # this will make it easier to update weights if we prune out rows at this level
	matching_rows = tables[0].indices[0].get(group, [])
	for m in matching_rows:
   	    weight_sums[0][m] = groupByCounts[group]
	print "%s, %d " % (group, groupByCounts[group])

    numGroupByKeys = len(groupByCounts.keys())
    
	#weights adjust for group distribution
	sub_weights = list((0 if s==0 else (1/(s*float(numGroupByKeys)))) for s in weight_sums[0])	
    
	#Uncomment below for uniform/regular WJ.
	sub_weights=list( 1 for w in weight_sums[0]) 
    
	sws = float(sum(sub_weights))
    sub_weights = list( w/sws for w in sub_weights)
    
    
    for i in xrange(n):
        p, groupByValue = sample_path_dynamic_weights_sum_groupby(failed_path,sub_weights,updateFunction, whereClauses, weight_sums,groupByCounts,numGroupByKeys,verbose)
	numSamplesPerGroup[groupByValue] +=1
	j = numSamplesPerGroup[groupByValue]
        sx[groupByValue] += p
        sxx[groupByValue] += p * p
	ex[groupByValue] = sx[groupByValue] / (float(j))
	exx[groupByValue] = sxx[groupByValue] / (float(j))
	if (sum(list(count[c] for c in count)))==7:
	    break
	v[groupByValue] = exx[groupByValue] - ex[groupByValue] * ex[groupByValue]
        hw[groupByValue] = 1.96 * (v[groupByValue] ** 0.5) / max((numSamplesPerGroup[groupByValue] ** 0.5),1)
	relError[groupByValue] = 0 if ex[groupByValue] ==0 else hw[groupByValue]/ex[groupByValue]
	numGroupsReachRelErrorCutoff = 0
	
	if relError[groupByValue]<=0:	
	    continue

	if relError[groupByValue]<=0.04 and relError[groupByValue]>0:
	    count[groupByValue]=1	
	    continue

	
	

	#print "%s, %s: %s,%s,%s,%s" % (count[groupByValue],(float(j)+1),p, ex[groupByValue],(exx[groupByValue]-(ex[groupByValue]*ex[groupByValue])),(1.96 * ((exx[groupByValue]-(ex[groupByValue]*ex[groupByValue])) ** 0.5) / ((j) ** 0.5))/(ex[groupByValue]+1)  )
    print "num total samples %s" %i
    # return 95% CI from large-sample confidence interval

    print "group: val+CI, val-CI, variance, rel error,samples per group"
    for group in tables[0].indices[0].keys():
        v[group] = exx[group] - ex[group] * ex[group]
        hw[group] = 1.96 * (v[group] ** 0.5) / max((numSamplesPerGroup[group] ** 0.5),1)
        relError[group] = 0 if ex[group] ==0 else hw[group]/ex[group]
	print "%s: %.2f, %.2f, %.2f, %.5f  %s"%( group, ex[group]-hw[group], ex[group]+hw[group], v[group], relError[group],numSamplesPerGroup[group])

    for j in xrange(0,len(tables),1):
        countFailed[j] = 0
        countFailedMult[j] = 0
        totalFailures[j] = 0
        for i in xrange(0,len(failed_path[j]),1):
	    totalFailures[j]+=failed_path[j][i]
            if failed_path[j][i] >0:
	        countFailed[j] +=1
            if failed_path[j][i] >1:
	        countFailedMult[j] +=1
        print "table %s, Total failed %f : %f Failed: %f multiple: %f" % (j,n,totalFailures[j],countFailed[j],countFailedMult[j])

#    for (s, c) in itertools.izip(weight_sums, weight_counts):
#        print " ",
#        for (sv, cv) in itertools.izip(s, c):	    
#            print ("%.2f" % (sv / cv)),
#	print
#    return ex - hw, ex + hw, v, relError, totalFailures[0], totalFailures[1],countFailedMult[0],countFailedMult[1]
   # return ex, hw, v, relError, totalFailures[0], totalFailures[1],countFailedMult[0],countFailedMult[1]



##################



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
    prefix = './data/flight/'
    
   
    numLines=0
    count=0
    flights = []
    f = open( prefix+'2008.csv', 'rU' ) #open the file in read universal mode
    next(f)
    for line in f:
        cells = line.split( "," )
		numLines+=1
	#tailnum, "1" to do counts using the sum method
        flights.append( [cells[10],1] )
    f.close()
    print "flights #rows %s",numLines

   

    numLines=0
    count=0
    totalQty = 0
    planes = []
    f = open( prefix+'plane-data.csv', 'rU' ) #open the file in read universal mode
    next(f)
    for line in f:
        cells = line.split( "," )
		if len(cells)==1:
			continue
		numLines+=1
		#engine_type, tailnum
		planes.append( [ cells[ 7 ], cells[ 0 ] ] )

    f.close()
    print "planes #rows %s",numLines


    global tbl2 
    tbl2 = [Table(planes),
        Table(flights)]    

    baseWeight=[1.0,1.0]

    print whereClauses
        
    print
    print " wander join sum with pruning"
    print groupby_wander_join_sum(tbl2,whereClauses,{},baseWeight,10000000,False)


None


 
 

