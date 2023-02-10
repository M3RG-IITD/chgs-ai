import numpy as np
import pandas  as pd
import re

'''
In a glass with 20% of As2S3 and 80% of As2Te3 then molar concentration As in total will (2/5 x 20%) + (3/5 x 80%) = 40% similarly 
For S = (3/5x20%) = 12% 
For Te = (3/5x80%) = 48% , Thus, the molar concentration for As, S, Te comes out to be 40, 12, 48 which is true as it is sum is 100 (40+12+48 =100). 
Dimension of the data depends on the number of elements present in a glass. Considering if there is a glass B with 4 elements say As, S, Te, Se with equal molar concentration of 25 each. Then our complete data would look like 

Glasses	As	S	Te	Se
Glass A	40	12	48	0
Glass B	25	25	25	25


'''

def reject_outliers(data, m = 2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        return (s < m)

def mean_dup(x_):
        global reject_outliers
        if 1==len(np.unique(x_.values)):
            return x_.values[0]
        else:
            x = x_.values[reject_outliers(x_.values.copy())]
            x_mean = x.mean()
            mask = (x_mean*0.975 <= x) & (x <= x_mean*1.025)
            return x[mask].mean()

def remove_duplicate(df):  
    '''
        Removes duplicates in dataframe and element samples whose composition is not 100%

        input format  ->  df  = dataframe 
    '''
    features = df.columns.values.tolist()
    features.remove(df.columns[-1])
    property_name = df.columns[-1] 

    df = df[df[features].sum(axis=1).between(99,101)]
    df = df.groupby(features,as_index=False).agg(mean_dup)
    df = df.dropna()
    df = df.loc[(df[property_name])> 0]
    return df

    
def create_chemicalDict(sa , feature):     # feature is a String  'TG'


    '''
    Return elements dictionary and nested dictionary of composition 



    format type -> 

            features =df.columns.values.tolist()
            features.remove('TG')
            
            final_dict , chem = create_chemicalDict(features, 'TG')
    '''

    final_dict = {}

    chem = {}
    for i in sa:
        if i == feature:
            final_dict[feature] == 0
        c = re.findall(r'([A-Z][a-z]?)(\d*)', i)
        dict = {}
        for x in c:

            if(x[1] == ''):
                dict[x[0]] = 1
            else:
                dict[x[0]] = int(x[1])
            final_dict[x[0]] = 0
        chem[i] = dict
    final_dict[feature] = 0
    
    return final_dict, chem

def totalSum(string, chem):
    val_dict = chem[string]
    sum_val = 0
    for i in val_dict:
        
        sum_val += val_dict[i]
    return sum_val

def computer(dict, final_dict, chem, property_name ):   # propert_name = 'TG'
    final_val = final_dict.copy()
    for i in dict:
        if (i == property_name):

            final_val[property_name] = dict[i]
            
        else:
            
            if dict[i] == 0:
                continue
            else :
                sum_val = totalSum(i, chem)
                comp_dict = chem[i]
                for j in comp_dict:
                    
                    final_val[j] += (dict[i] * comp_dict[j])/ sum_val
          
    return final_val

def data_creator(List_dict, final_dict, chem ,  property_name_val):

    '''
        Returns a combined dataframe formed by appending feature dictionary from 
        computer() 

        input format -> List_dict = dictionary of raw data frame in records form 
        use code "d = df.to_dict(orient='records')"

        format type -> 
        final_dict , chem = create_chemicalDict(features, 'TG')
        data_creator(d,final_dict, chem, 'TG')

    '''

    output_df = pd.DataFrame()
    
    for i in List_dict:
        final_val = computer(i,final_dict, chem, property_name_val)
        output_df = output_df.append(final_val, ignore_index=True)

    # column_to_move = output_df.pop(property_name_val)
    # output_df.insert(len(output_df.columns), property_name_val, column_to_move)

    col_list = output_df.columns.tolist()  
    col_list.remove(property_name_val)
    col_list = col_list + [property_name_val]
    output_df = output_df[col_list]

    output_df = remove_duplicate(output_df)
    
    return output_df





def remove_minSamples(df, count = 20):

    '''
        Returns dataframe with atleast 20 samples in features   
        input format -> df = dataframe 
    '''
    hist_data2 = df.astype(bool).sum(axis = 0)
    mask = hist_data2 >= count 
    df = df[list(hist_data2[mask].index)]


    df = remove_duplicate(df)

    return df
    

















############################## PLOTS ################################

def circular_plot(df):
    
    xx2 = df[df.columns[:-1]]
    hist_data4 = xx2.astype(bool).sum(axis=0)
    hist_data4.value_counts()
    all_dict = hist_data4.to_dict()
    all_frame = hist_data4.to_frame(name = 'value')
    keys = all_dict.keys()
    values = all_dict.values()

    # Reorder the dataframe

    # initialize the figure
    plt.figure(figsize=(220,103))
    ax = plt.subplot(111, polar=True)
    plt.axis('off')


    # Constants = parameters controling the plot layout:
    upperLimit = 1000
    lowerLimit = 300
    labelPadding = 4

    all_frame = all_frame.sort_values(by=['value'])
    # Compute max and min in the dataset
    max = all_frame['value'].max()

    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
    slope = (max - lowerLimit) / max
    heights = slope * all_frame.value + lowerLimit

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi / len(all_frame.index)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(all_frame.index)+1))
    angles = [element * width for element in indexes]
    angles

    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width, 
        bottom=lowerLimit,
        linewidth=2, 
        edgecolor="white",
        color="#61a4b2",
    
    )

    # Add labels
    for bar, angle, height, label, v in zip(bars,angles, heights, all_frame.index, all_frame['value']):

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi/2 and angle < 3*np.pi/2:
            alignment = "right"
            rotation = rotation + 180
        else: 
            alignment = "left"

    # Finally add the labels
        ax.text(
            x=angle, 
            y=lowerLimit + bar.get_height() + labelPadding, 
            s="  " +label + "   " + str(v)+ "    " , 
            ha=alignment, 
            va='center', 
            rotation=rotation, 
            rotation_mode="anchor") 
    plt.savefig('circularPlot_minSample.png')

