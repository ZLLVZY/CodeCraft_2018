import math
import datetime

class ecs(object):          #create ecs class
    def __init__(self,norm,date):
        self.norm=norm
        self.date=date
        
class flavor(object):                       #create flvaor class
    def __init__(self,name,cpu,mem,parameters):
        self.name=name
        self.cpu=cpu
        self.mem=mem
        self.parameters=parameters
        
def get_prdata(input_lines):            #get predict_data from input
    F=[]
    createVar ={}
    j=1
    for item_input in input_lines:      #get predict_flavor
        if item_input[0]=='f':
            temp=''.join(item_input).split( )
            createVar["f%s" % j]=flavor(temp[0],int(temp[1]),int(temp[2])/1024,int(temp[2])/1024/int(temp[1]))
            F.append(createVar['f%s' % j])
            j+=1
            
        if item_input[0]=='C':
            d='CPU'
        if item_input[0]=='M':
            d='MEM'
            
    T=[input_lines[-2][0:19],input_lines[-1][0:19]]    #get predict_time
    
    temp=''.join(input_lines[0]).split( )    #get server data
    S=[temp[0],temp[1]]
    return T,F,S,d

def feature1(ecs_lines,F):              #get flavor norm from train_data
    E=[]
    createVar ={}
    j=1
    for item_ecs in ecs_lines:
        temp=''.join(item_ecs).split( )
        for i in F:
            if temp[1]==i.name:
                createVar["e%s" % j]=ecs(temp[1],temp[2]+' '+temp[3])
                E.append(createVar["e%s" % j])
                j+=1
    return E

def get_tstart(ecs_lines):         #get first time from train data
    temp=''.join(ecs_lines[0]).split( )
    tstart=temp[2]+' '+temp[3]
    return tstart

def get_interval_num(t_prbegin,t_prend,tstart):    #get t_prend-t_prbegin&delta
    delta_pr=t_prend-t_prbegin
    delta=t_prbegin-tstart
    interval_num=delta.days//delta_pr.days
    if interval_num<1000:
        return interval_num,delta_pr
    else:
        return 1000,delta_pr

def get_interval2(t_prbegin,interval_num,delta_pr,E,name):        #get F.name=name ''s feature_data
    L=[]
    for i in range(interval_num,0,-1):
        k=0
        for j in E:
            temp=datetime.datetime.strptime(j.date,'%Y-%m-%d %H:%M:%S')
            if (((t_prbegin-delta_pr*i)<temp)and(temp<t_prbegin-delta_pr*(i-1))and(j.norm==name)):
                k+=1
        L.append(k)
    return L

def get_interval(t_prbegin,interval_num,delta_pr,E,F):              #get all feature_data
    createVar ={}
    k=0
    I=[]
    for i in F:
        createVar["i%s" % k]=get_interval2(t_prbegin,interval_num,delta_pr,E,i.name)
        I.append(createVar["i%s" % k])
        k+=1
    return I

def train(train_data,interval_num):                 #Adagrad train model:   y=w1x1+w2x2+b
    j=interval_num-2
    lr=0.0000015                                                     #learning_rate
    train_num=8000                                  #trian_num
    w1,w2,b=0.9,0.45,0.5                                 #weight&bias
    lr_w1,lr_w2,lr_b=0,0,0
    for n in range(interval_num):                     #if train_data.all=0,train_data+=0.0000000000001
        if train_data[n]==0:
            train_data[n]+=0.000000000001
            
    for i in range(train_num):                          #start train
        w1_grad,w2_grad,b_grad=0,0,0
        for k in range(j):
            w1_grad=w1_grad-2*train_data[k]*(train_data[k+2]-w1*train_data[k]-w2*train_data[k+1]-b)
            w2_grad=w2_grad-2*train_data[k+1]*(train_data[k+2]-w1*train_data[k]-w2*train_data[k+1]-b)
            b_grad=b_grad-2*(train_data[k+2]-w1*train_data[k]-w2*train_data[k+1]-b)
            
        w1_grad=w1_grad/j               #calculate grade
        w2_grad=w2_grad/j
        b_grad=b_grad/j
        
        lr_w1=lr_w1+w1_grad**2    
        lr_w2=lr_w2+w2_grad**2
        lr_b=lr_b+b_grad**2
        
        w1=w1-w1_grad*lr/(lr_w1**0.5)                    #update parameters
        w2=w2-w2_grad*lr/(lr_w2**0.5)
        b=b-b_grad*lr/(lr_b**0.5)
    
    print(w1,w2,b)
    y=round(w1*train_data[-2]+w2*train_data[-1]+b)           
    y=int(y)
    if y<0:
        y=0
    return y
    
def train_all(N,interval_num):                #train every_flavor's model
    result_pr=[]
    for i in range(len(N)):
        result_pr.append(train(N[i],interval_num))
    return result_pr                     #get predict result

def put2(s_num,f,S,F):                    #put a flavor in server
    temp=1
    k=len(s_num)
    createVar ={}
    for i in range(len(s_num)):
        if (f.cpu<=s_num[i]['cpu'])and(f.mem<=s_num[i]['mem']):
            s_num[i]['cpu']=s_num[i]['cpu']-f.cpu
            s_num[i]['mem']=s_num[i]['mem']-f.mem
            s_num[i][f.name]=s_num[i][f.name]+1
            temp=0
            break
    if temp==1:
        k+=1
        createVar['s%s' % k]={}
        createVar['s%s' % k]['cpu']=int(S[0])-f.cpu
        createVar['s%s' % k]['mem']=int(S[1])-f.mem
        for i in F:
            createVar['s%s' % k][i.name]=0
        createVar['s%s' % k][f.name]=1
        s_num.append(createVar['s%s' % k])
    return s_num
            
def put(result_pr,S,F):           #put all flavor in server
    s1={}
    s1['cpu']=int(S[0])
    s1['mem']=int(S[1])
    for i in F:
        s1[i.name]=0
    s_num=[s1]
   # for i in range(-1,-len(result_pr)-1,-1):
    for i in range(len(result_pr)):
        for j in range(result_pr[i]):
            s_num=put2(s_num,F[i],S,F)
    return s_num

    
def get_result(result,s_num,F,result_pr):
    f_num=0
    for i in result_pr:
        f_num+=i
    result.append(f_num)
        
    for i in range(len(F)):
        result.append(F[i].name+' '+str(result_pr[i]))
    result.append('')
    result.append(len(s_num))
    for i in range(len(s_num)):
        result.append(str(i+1)+' ')
        for j in range(len(F)):
            if s_num[i][F[j].name]!=0:
                result[-1]=result[-1]+F[j].name+' '+str(s_num[i][F[j].name])+' '
    return result

def feature2(N):
    for i in range(len(N)):
        for k in range(len(N)-2):
            if (N[i][k+1]-N[i][k]>3)and(N[i][k+1]-N[i][k+2]>3):
                N[i][k+1]-=2
            if (N[i][k+1]-N[i][k]<3)and(N[i][k+1]-N[i][k+2]<3):
                N[i][k+1]+=2
            if (N[i][k+1]-N[i][k]>5)and(N[i][k+1]-N[i][k+2]>5):
                N[i][k+1]-=4
            if (N[i][k+1]-N[i][k]<5)and(N[i][k+1]-N[i][k+2]<5):
                N[i][k+1]+=4
    return N

def put_feature(d,F):
    for i in range(len(F)):
        for j in range(i+1,len(F)):
            if d=='CPU':
                if F[i].parameters>F[j].parameters:
                     F[i], F[j] = F[j], F[i]
            if d=='MEM':
                if F[i].parameters<F[j].parameters:
                     F[i], F[j] = F[j], F[i]
    return F
    
def seq(input_lines,F,result_pr):
    F2=[]
    result_pr2=[]
    for item_input in input_lines:      
        if item_input[0]=='f':
            temp=''.join(item_input).split( )
            for i in range(len(F)):
                if F[i].name==temp[0]:
                    F2.append(F[i])
                    result_pr2.append(result_pr[i])
    return F2,result_pr2

def predict_vm(ecs_lines, input_lines):
    # Do your work from here#
    T,F,S,d=get_prdata(input_lines)    #T:pr_time   F:flavor,cpu,mem  S:server's cpu,men
    F=put_feature(d,F)
    E=feature1(ecs_lines,F)              #E:train_data's:flavor,time
    tstart=get_tstart(ecs_lines)
    tstart=datetime.datetime.strptime(tstart,'%Y-%m-%d %H:%M:%S')
    t_prbegin=datetime.datetime.strptime(T[0],'%Y-%m-%d %H:%M:%S')
    t_prend=datetime.datetime.strptime(T[1],'%Y-%m-%d %H:%M:%S')
    interval_num,delta_pr=get_interval_num(t_prbegin,t_prend,tstart)
    N=get_interval(t_prbegin,interval_num,delta_pr,E,F)        #N:train_data(feature)
    N=feature2(N)
    result_pr=train_all(N,interval_num)
    s_num=put(result_pr,S,F)
    F,result_pr=seq(input_lines,F,result_pr)
    result = []
    result=get_result(result,s_num,F,result_pr)
    if ecs_lines is None:
        print 'ecs information is none'
        return result
    if input_lines is None:
        print 'input file information is none'
        return result
    return result
