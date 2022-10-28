import numpy as np
import scipy.spatial.distance
from collections import deque
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tabulate
from sklearn import preprocessing
from joblib import load
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MaxAbsScaler

# spatial threshold or the dispersion itself
max_dispersion = np.deg2rad(1.6)
# temporal threshold or duration
min_duration = 100

cl = load('classifier.joblib') 

def vector_dispersion(vectors):
    distances = scipy.spatial.distance.pdist(vectors, metric='cosine')
    distances.sort()
    cut_off = np.max([distances.shape[0] // 5, 4])
    return np.arccos(1. - distances[-cut_off:].mean())

def gaze_dispersion(eye_data):
    base_data = eye_data

    vectors = []
    for p in base_data:
        vectors.append((p['gazeDirection_x'], p['gazeDirection_y'], p['gazeDirection_z']))
    vectors = np.array(vectors, dtype=np.float32)

    if len(vectors) < 2:
        return float("inf")
    else:
        return vector_dispersion(vectors)  

def get_centroid(eye_data):
    '''Calculates the centroid for each point in a df of points.
    Input: Df of points.
    Output: Vector containg the centroid of all points.'''
    x = [p['gazeDirection_x'] for p in eye_data]
    y = [p['gazeDirection_y'] for p in eye_data]
    z = [p['gazeDirection_z'] for p in eye_data]
    return (sum(x) / len(eye_data), sum(y) / len(eye_data), sum(z) / len(eye_data))


def detect_fixations(gaze_data):
    # Convert Pandas data frame to list of Python dictionaries
    gaze_data = gaze_data.T.to_dict().values()

    candidate = deque()
    future_data = deque(gaze_data)
    while future_data:
        # check if candidate contains enough data
        if len(candidate) < 2 or candidate[-1]['eyeDataTimestamp'] - candidate[0]['eyeDataTimestamp'] < min_duration:
            datum = future_data.popleft()
            candidate.append(datum)
            continue

        # Minimal duration reached, check for fixation
        dispersion = gaze_dispersion(candidate)
        if dispersion > max_dispersion:
            # not a fixation, move forward
            candidate.popleft()
            continue

        # Minimal fixation found. Try to extend!
        while future_data:
            datum = future_data[0]
            candidate.append(datum)

            dispersion = gaze_dispersion(candidate)
            if dispersion > max_dispersion:
                # end of fixation found
                candidate.pop()
                break
            else:
                # still a fixation, continue extending
                future_data.popleft()
        centroid = get_centroid(candidate)
        yield {"start": candidate[0]['eyeDataTimestamp'], "end": candidate[-1]['eyeDataTimestamp'],
               "duration": candidate[-1]['eyeDataTimestamp'] - candidate[0]['eyeDataTimestamp'],
              "centroid": centroid, "dispersion": dispersion}
        candidate.clear()


def only_valid_data(data):
    '''Returns only valid gaze points. Those have values in gazeDirection_x etc.'''
    return data[(data.gazeHasValue == True) & (data.isCalibrationValid == True)]

def calculate_fixation_features(df_fixations, timespan):
    '''Calculates the fixation features. 
    Input: Dataframe with fixation, timespan of data chunk in seconds.
    Output: Dict containing the fixation features.'''

    min_fix = df_fixations["duration"].min()
    max_fix = df_fixations["duration"].max()
    mean_fix = df_fixations["duration"].mean()
    var_fix = df_fixations["duration"].var()
    std_fix = df_fixations["duration"].std()
    
    min_dispersion = df_fixations["dispersion"].min()
    max_dispersion = df_fixations["dispersion"].max()
    mean_dispersion = df_fixations["dispersion"].mean()
    var_dispersion = df_fixations["dispersion"].var()
    std_dispersion = df_fixations["dispersion"].std()
    
    fixation_frequency_second = (len(df_fixations["dispersion"]) / timespan)
    
    # print("min: ", min_fix, " max: ", max_fix, " mean: ", mean_fix, " var: ", var_fix, "std: ", std_fix)
    # print("min dispersion: ", min_dispersion, " max: ", max_dispersion, " mean: ", mean_dispersion, 
    #      " var: ", var_dispersion)
    #print("x dispersion: ", dispersion_x, " y dispersion: ", dispersion_y, " z dispersion: ", dispersion_z)
    
    return {"meanFix": mean_fix, "minFix": min_fix, "maxFix": max_fix, "varFix": var_fix, "stdFix": std_fix,
            "meanDis": mean_dispersion, "minDis": min_dispersion, "maxDis": max_dispersion,
            "varDis": var_dispersion, "stdDisp": std_dispersion,
            "freqDisPerSec": fixation_frequency_second}
    

def get_fixation_df(df_valid):
    '''Calls function to calculate Fixations. Converts the list of fixations to a dataframe and numbers the rows as index.
     Input: Dataframe containg valid gaze points.
     Output: Dataframe containing the fixation features.'''
    fixations = list(detect_fixations(df_valid))
    df = pd.DataFrame(fixations)
    df['index'] = range(1, len(df) + 1)
    # df.head()
    return df

def calculate_directions_of_list(points):
    '''Calculates the dominant direction of points.
    Input: Dataframe containing fixation points.
    Output: Dict with dominant direction for x (xDir) and y (yDir).
    '''
    x_values, y_values, z_values = zip(*points['centroid'])
    # Get a list of whether a given value is greater then the previous one in the list
    res_x = [float(val1) < float(val2) for val1, val2 in zip(x_values, x_values[1:])]
    # Sum all that are True
    sum_x = sum(res_x)
    # Divide the sum by the total number of values to get the desired output.
    # dir_x is -1 if there are no fixation (i.e. prevent division by zero)
    dir_x = -1
    if len(res_x) != 0:
        dir_x = sum_x/len(res_x)
        
    res_y = [float(val1) < float(val2) for val1, val2 in zip(y_values, y_values[1:])]
    sum_y = sum(res_y)
    dir_y = -1
    if len(res_y) != 0:
        dir_y = sum_y /len(res_y)
      
    
    return {"xDir": dir_x, "yDir": dir_y}

def calculate_fixation_density(df_all, df_fix):
    '''Calculates the fixation density per area.
    Input: Dataframe with all valid gazepoints, Dataframe with fixations.
    Output: Dict containing the fixation density.'''
    min_x = df_all['gazeDirection_x'].min()
    min_y = df_all['gazeDirection_y'].min()
    max_x = df_all['gazeDirection_x'].max()
    max_y = df_all['gazeDirection_y'].max()
    
    length = abs(max_x-min_x)
    height = abs(max_x-min_x)
    area = length*height
    
    number_of_fixations = len(df_fix)
    
    fix_dens = -1
    if area != 0: 
        fix_dens = number_of_fixations/area
    return {"fixDensPerBB": fix_dens}


def get_features(df, timespan=20):
    ''' Calculates the features for a raw gaze data in chunks of n seconds.
    Input: Dataframe with raw gaze data, timespan to chunk, label (i.e. activity class).
    Output: List of dictionaries, one dictionary contains a chunk of features.
    '''
    
    list_of_features = []
    i = 0
    while i < len(df)-1: 
        newdf = pd.DataFrame(columns=df.columns)
        start_time = df["eyeDataTimestamp"][i]  
        
        while i < len(df)-1 and df["eyeDataTimestamp"][i] < (start_time+timespan*1000):
            entry = df.iloc[[i]]
            newdf = pd.concat([newdf,entry])
            i+=1
        newdf.reset_index(inplace = True)
        
        if (len(newdf) > timespan*28):
            newdf_valid = only_valid_data(newdf)
            df_fixations = get_fixation_df(newdf_valid)
            
            features = calculate_fixation_features(df_fixations, timespan)

            directions = calculate_directions_of_list(df_fixations)            
            density = calculate_fixation_density(newdf_valid, df_fixations)
            
            features.update(directions)
            features.update(density) 
            list_of_features.append(features)
                
            return list_of_features
        else:
            print("Too few points")
            return []

def classify(df):
    features = get_features(df)
    if (len(features) > 0):
        features = features[0]
        test_features = np.array([features["meanFix"], features["maxFix"], features["varFix"], features["xDir"], features["yDir"], features["fixDensPerBB"]])
        test_features = test_features.reshape(1, -1)
        scaler = MaxAbsScaler()
        scaler.fit(test_features)
        scaled = scaler.transform(test_features)
        pred = cl.predict(test_features)
        return pred
    else:
        return "Too few points"

if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    classify(df)