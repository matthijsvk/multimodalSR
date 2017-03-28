from collections import defaultdict
import random
a,b,k,T=10,10,10,30 
p_ctc=defaultdict(float)
sig1=[chr(i) for i in xrange(97,123)]
sig1+=['_',' ']

#============Test values for CTC and CLM============================
for i in sig1:
	p_ctc[i]=[0]*(T+1) #Make p_ctc a 1-d array
	for j in xrange(T+1):
		p_ctc[i][j]=random.random()

p_clm=defaultdict(float)
for i in sig1:
	p_clm[i]=random.random()
#===================================================================

p_b=defaultdict(float)
p_nb=defaultdict(float)
z_prev=[]
z_prev.append('_')
p_b['_']=[0]*(T+1)
p_nb['_']=[0]*(T+1)
p_b['_'][0],p_nb['_'][0]=1,0
sig=[chr(i) for i in xrange(97,123)]
sig.append(' ')


import heapq
for t in xrange(1,T+1): #For every time instant
    #=================
    #Call the CTC here
    #=================
    z_next=[]
    for l in z_prev: #Every string in previous set 

        #=================
        #Call the CLM here
        #=================

    	# Initializing all probabilities of string 'l' in p_b and p_nb
    	if l!='_':
        	p_b[l]=[0]*(T+1)
        	p_nb[l]=[0]*(T+1)
        
        l_dash = l[:-1]
        p_b[l_dash]=[0]*(T+1)
        p_nb[l_dash]=[0]*(T+1)

        # l is followed by a 'blank'
        
        p_b[l][t] = p_ctc['_'][t]*(p_b[l_dash][t-1]+p_nb[l_dash][t-1])
        
        # l is followed by last character of l
        p_nb[l][t] = p_ctc[l[-1]][t]*(p_nb[l_dash][t-1])

        # loop for remaining characters i.e., excluding 'blank'
        l_plus=''
        for c in sig:

            l_plus=l+c

            # Initializing all probabilities of string 'l' in p_b and p_nb
            p_b[l_plus]=[0]*(T+1)
            p_nb[l_plus]=[0]*(T+1)
            
            if c!=l[-1]:
            	# c is not equal to last character of l
                p_nb[l][t]=p_ctc[c][t]*((p_clm[c])**a)*(p_b[l][t-1]+p_nb[l][t-1])

            else:

                p_nb[l][t]=p_ctc[c][t]*((p_clm[c])**a)*(p_b[l][t-1])

            z_next.append(l_plus)
    c=[]
    for i in z_next:
        c.append((p_b[i][t]+p_nb[i][t]))
    x=heapq.nlargest(k,c)
    z_prev=[z_next[i] for i in x]
d=[]
for i in z_prev:
    d.append((p_b[i][T]+p_nb[i][T]))
ind=d.index(max(d))
ans=z_prev[ind]
print ans