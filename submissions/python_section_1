from typing import Dict, List
from collections import defaultdict
from itertools import permutations
import re
import pandas as pd
from geopy.distance import geodesic
import polyline


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
        for i in range(0,len(lst),n):
        lst[i:i+n]=reversed(lst[i:i+n])
    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    length_dict=defaultdict(list)
    
    for s in lst:
        length_dict[len(s)].append(s)
    
    return dict(length_dict)

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    dict = {}

    def flatten(current, parent_key=''):
        for key, value in current.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                flatten(value, new_key)
            else:
                dict[new_key] = value

    flatten(nested_dict)
    return dict

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    return list(map(list, set(permutations(nums))))


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    patterns = r'(\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{4}\.\d{2}\.\d{2}\b)'
    return re.findall(patterns, text)
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coords = polyline.decode(polyline_str)
    distances = [0.0] + [geodesic(coords[i-1], coords[i]).meters for i in range(1, len(coords))]
    return pd.DataFrame(coords, columns=['latitude', 'longitude']).assign(distance=distances)


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    transformed_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            original_value = rotated_matrix[i][j]
            original_index_sum = (n - j - 1) + i
            row.append(original_value * original_index_sum)
        transformed_matrix.append(row)
    
    return transformed_matrix


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    def check_completeness(group):
        start_time = group['timestamp'].min().normalize()
        full_range = pd.date_range(start=start_time, periods=168, freq='H')
        return full_range.isin(group['timestamp']).all()

    return df.groupby(['id', 'id_2']).apply(check_completeness)
