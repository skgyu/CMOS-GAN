import os


def get_id_from_loc(locstr):
    
    return int(os.path.split(locstr)[1].split('.')[0].split('_')[0] )
  

    
def sort_by_name(locstr):

    ret=os.path.split(locstr)[1].split('.')[0].split('_')
    ans=[]

    for x in ret:
        if x.isdigit():
            ans.append( int(x) )


    return  tuple(ans)

    
