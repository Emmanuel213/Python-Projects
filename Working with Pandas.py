{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "city_dictionary = {'Chicago':4567898,\n",
    "                   'New Jersey':59876543,\n",
    "                   'Mexico':23347654,\n",
    "                   'Miami':65438976}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "city = pd.Series(city_dictionary)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Chicago        4567898\n",
       "New Jersey    59876543\n",
       "Mexico        23347654\n",
       "Miami         65438976\n",
       "dtype: int64"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "city"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "New Jersey    59876543\n",
       "Mexico        23347654\n",
       "Miami         65438976\n",
       "dtype: int64"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "city['New Jersey':'Miami']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "a    1\n",
       "b    2\n",
       "c    3\n",
       "d    4\n",
       "dtype: int64"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.Series({'a':1,'b':2,'c':3,'d':4})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,\n",
    "'Florida': 170312, 'Illinois': 149995}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "frame = pd.Series(area_dict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "California    423967\n",
       "Texas         695662\n",
       "New York      141297\n",
       "Florida       170312\n",
       "Illinois      149995\n",
       "dtype: int64"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "frame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_frame = pd.DataFrame(frame, columns=['population'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>population</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>California</th>\n",
       "      <td>423967</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Texas</th>\n",
       "      <td>695662</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>New York</th>\n",
       "      <td>141297</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Florida</th>\n",
       "      <td>170312</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Illinois</th>\n",
       "      <td>149995</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            population\n",
       "California      423967\n",
       "Texas           695662\n",
       "New York        141297\n",
       "Florida         170312\n",
       "Illinois        149995"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_frame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['population'], dtype='object')"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_frame.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['California', 'Texas', 'New York', 'Florida', 'Illinois'], dtype='object')"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_frame.index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "California    423967\n",
       "Texas         695662\n",
       "New York      141297\n",
       "Florida       170312\n",
       "Illinois      149995\n",
       "Name: population, dtype: int64"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_frame['population']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "###Working with the pandas index\n",
    "index = pd.Index([1,2,3,4])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Int64Index([1, 2, 3, 4], dtype='int64')"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Int64Index([3, 4], dtype='int64')"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "index[2:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "##Performing index operations\n",
    "#Indexing in pandas allows you to also perform set arithmetic\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "indA = pd.Index([1,2,3,4,5])\n",
    "indB = pd.Index([3,4,5,6,7])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Int64Index([3, 4, 5], dtype='int64')"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "indA & indB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Int64Index([1, 2, 6, 7], dtype='int64')"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "indA ^ indB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Int64Index([1, 2, 3, 4, 5, 6, 7], dtype='int64')"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "indA | indB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "###Working with implicit and explicit indexing\n",
    "###First we would look at series indexing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "ind_case1 = pd.Series(['a','b','c','d'], index=[1,2,3,4])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    a\n",
       "2    b\n",
       "3    c\n",
       "4    d\n",
       "dtype: object"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ind_case1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3    c\n",
       "4    d\n",
       "dtype: object"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ind_case1[2:4]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "###Using loc, iloc, ix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    a\n",
       "2    b\n",
       "3    c\n",
       "dtype: object"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ind_case1.loc[1:3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [],
   "source": [
    "##loc is used for exlicitly defined indexes\n",
    "##iloc is used for implicitly defined indexes\n",
    "##ix is a hybrid of both of them"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "##Indexing and Slicing in Data frames\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [],
   "source": [
    "area = pd.Series({'California': 423967, 'Texas': 695662,\n",
    "                'New York': 141297, 'Florida': 170312,\n",
    "                'Illinois': 149995})\n",
    "\n",
    "pop = pd.Series({'California': 38332521, 'Texas': 26448193,\n",
    "                'New York': 19651127, 'Florida': 19552860,\n",
    "                'Illinois': 12882135})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.DataFrame({'Area':area, 'Population':pop})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Area</th>\n",
       "      <th>Population</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>California</th>\n",
       "      <td>423967</td>\n",
       "      <td>38332521</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Texas</th>\n",
       "      <td>695662</td>\n",
       "      <td>26448193</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>New York</th>\n",
       "      <td>141297</td>\n",
       "      <td>19651127</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Florida</th>\n",
       "      <td>170312</td>\n",
       "      <td>19552860</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Illinois</th>\n",
       "      <td>149995</td>\n",
       "      <td>12882135</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Area  Population\n",
       "California  423967    38332521\n",
       "Texas       695662    26448193\n",
       "New York    141297    19651127\n",
       "Florida     170312    19552860\n",
       "Illinois    149995    12882135"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Density'] = df['Population'] / df['Area']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Area</th>\n",
       "      <th>Population</th>\n",
       "      <th>Density</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>California</th>\n",
       "      <td>423967</td>\n",
       "      <td>38332521</td>\n",
       "      <td>90.413926</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Texas</th>\n",
       "      <td>695662</td>\n",
       "      <td>26448193</td>\n",
       "      <td>38.018740</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>New York</th>\n",
       "      <td>141297</td>\n",
       "      <td>19651127</td>\n",
       "      <td>139.076746</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Florida</th>\n",
       "      <td>170312</td>\n",
       "      <td>19552860</td>\n",
       "      <td>114.806121</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Illinois</th>\n",
       "      <td>149995</td>\n",
       "      <td>12882135</td>\n",
       "      <td>85.883763</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Area  Population     Density\n",
       "California  423967    38332521   90.413926\n",
       "Texas       695662    26448193   38.018740\n",
       "New York    141297    19651127  139.076746\n",
       "Florida     170312    19552860  114.806121\n",
       "Illinois    149995    12882135   85.883763"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>California</th>\n",
       "      <th>Texas</th>\n",
       "      <th>New York</th>\n",
       "      <th>Florida</th>\n",
       "      <th>Illinois</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Area</th>\n",
       "      <td>4.239670e+05</td>\n",
       "      <td>6.956620e+05</td>\n",
       "      <td>1.412970e+05</td>\n",
       "      <td>1.703120e+05</td>\n",
       "      <td>1.499950e+05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Population</th>\n",
       "      <td>3.833252e+07</td>\n",
       "      <td>2.644819e+07</td>\n",
       "      <td>1.965113e+07</td>\n",
       "      <td>1.955286e+07</td>\n",
       "      <td>1.288214e+07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Density</th>\n",
       "      <td>9.041393e+01</td>\n",
       "      <td>3.801874e+01</td>\n",
       "      <td>1.390767e+02</td>\n",
       "      <td>1.148061e+02</td>\n",
       "      <td>8.588376e+01</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              California         Texas      New York       Florida  \\\n",
       "Area        4.239670e+05  6.956620e+05  1.412970e+05  1.703120e+05   \n",
       "Population  3.833252e+07  2.644819e+07  1.965113e+07  1.955286e+07   \n",
       "Density     9.041393e+01  3.801874e+01  1.390767e+02  1.148061e+02   \n",
       "\n",
       "                Illinois  \n",
       "Area        1.499950e+05  \n",
       "Population  1.288214e+07  \n",
       "Density     8.588376e+01  "
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Area</th>\n",
       "      <th>Population</th>\n",
       "      <th>Density</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>California</th>\n",
       "      <td>423967</td>\n",
       "      <td>38332521</td>\n",
       "      <td>90.413926</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Texas</th>\n",
       "      <td>695662</td>\n",
       "      <td>26448193</td>\n",
       "      <td>38.018740</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>New York</th>\n",
       "      <td>141297</td>\n",
       "      <td>19651127</td>\n",
       "      <td>139.076746</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Florida</th>\n",
       "      <td>170312</td>\n",
       "      <td>19552860</td>\n",
       "      <td>114.806121</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Illinois</th>\n",
       "      <td>149995</td>\n",
       "      <td>12882135</td>\n",
       "      <td>85.883763</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Area  Population     Density\n",
       "California  423967    38332521   90.413926\n",
       "Texas       695662    26448193   38.018740\n",
       "New York    141297    19651127  139.076746\n",
       "Florida     170312    19552860  114.806121\n",
       "Illinois    149995    12882135   85.883763"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "California    423967\n",
       "Texas         695662\n",
       "New York      141297\n",
       "Florida       170312\n",
       "Illinois      149995\n",
       "Name: Area, dtype: int64"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Area']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "California     90.413926\n",
       "Texas          38.018740\n",
       "New York      139.076746\n",
       "Florida       114.806121\n",
       "Illinois       85.883763\n",
       "Name: Density, dtype: float64"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Density']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "California    423967\n",
       "Texas         695662\n",
       "New York      141297\n",
       "Florida       170312\n",
       "Illinois      149995\n",
       "Name: Area, dtype: int64"
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Area"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[4.23967000e+05, 3.83325210e+07, 9.04139261e+01],\n",
       "       [6.95662000e+05, 2.64481930e+07, 3.80187404e+01],\n",
       "       [1.41297000e+05, 1.96511270e+07, 1.39076746e+02],\n",
       "       [1.70312000e+05, 1.95528600e+07, 1.14806121e+02],\n",
       "       [1.49995000e+05, 1.28821350e+07, 8.58837628e+01]])"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Area</th>\n",
       "      <th>Population</th>\n",
       "      <th>Density</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>California</th>\n",
       "      <td>423967</td>\n",
       "      <td>38332521</td>\n",
       "      <td>90.413926</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Area  Population    Density\n",
       "California  423967    38332521  90.413926"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.loc[:'California', : 'Density']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Area</th>\n",
       "      <th>Population</th>\n",
       "      <th>Density</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>California</th>\n",
       "      <td>423967</td>\n",
       "      <td>38332521</td>\n",
       "      <td>90.413926</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Texas</th>\n",
       "      <td>695662</td>\n",
       "      <td>26448193</td>\n",
       "      <td>38.018740</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>New York</th>\n",
       "      <td>141297</td>\n",
       "      <td>19651127</td>\n",
       "      <td>139.076746</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Florida</th>\n",
       "      <td>170312</td>\n",
       "      <td>19552860</td>\n",
       "      <td>114.806121</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Illinois</th>\n",
       "      <td>149995</td>\n",
       "      <td>12882135</td>\n",
       "      <td>85.883763</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Area  Population     Density\n",
       "California  423967    38332521   90.413926\n",
       "Texas       695662    26448193   38.018740\n",
       "New York    141297    19651127  139.076746\n",
       "Florida     170312    19552860  114.806121\n",
       "Illinois    149995    12882135   85.883763"
      ]
     },
     "execution_count": 87,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Area</th>\n",
       "      <th>Population</th>\n",
       "      <th>Density</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>California</th>\n",
       "      <td>423967</td>\n",
       "      <td>38332521</td>\n",
       "      <td>90.413926</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Texas</th>\n",
       "      <td>695662</td>\n",
       "      <td>26448193</td>\n",
       "      <td>38.018740</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>New York</th>\n",
       "      <td>141297</td>\n",
       "      <td>19651127</td>\n",
       "      <td>139.076746</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Area  Population     Density\n",
       "California  423967    38332521   90.413926\n",
       "Texas       695662    26448193   38.018740\n",
       "New York    141297    19651127  139.076746"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.iloc[:3, :3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.iloc[0,2] = 50"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Area</th>\n",
       "      <th>Population</th>\n",
       "      <th>Density</th>\n",
       "      <th>(3, 2)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>California</th>\n",
       "      <td>423967</td>\n",
       "      <td>38332521</td>\n",
       "      <td>50.000000</td>\n",
       "      <td>79</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Texas</th>\n",
       "      <td>695662</td>\n",
       "      <td>26448193</td>\n",
       "      <td>38.018740</td>\n",
       "      <td>79</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>New York</th>\n",
       "      <td>141297</td>\n",
       "      <td>19651127</td>\n",
       "      <td>139.076746</td>\n",
       "      <td>79</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Florida</th>\n",
       "      <td>170312</td>\n",
       "      <td>19552860</td>\n",
       "      <td>114.806121</td>\n",
       "      <td>79</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Illinois</th>\n",
       "      <td>149995</td>\n",
       "      <td>12882135</td>\n",
       "      <td>85.883763</td>\n",
       "      <td>79</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Area  Population     Density  (3, 2)\n",
       "California  423967    38332521   50.000000      79\n",
       "Texas       695662    26448193   38.018740      79\n",
       "New York    141297    19651127  139.076746      79\n",
       "Florida     170312    19552860  114.806121      79\n",
       "Illinois    149995    12882135   85.883763      79"
      ]
     },
     "execution_count": 95,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [],
   "source": [
    "d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "California    79\n",
       "Texas         79\n",
       "New York      79\n",
       "Florida       79\n",
       "Illinois      79\n",
       "Name: (3, 2), dtype: int64"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.pop((3,2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Area</th>\n",
       "      <th>Population</th>\n",
       "      <th>Density</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>California</th>\n",
       "      <td>423967</td>\n",
       "      <td>38332521</td>\n",
       "      <td>50.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Texas</th>\n",
       "      <td>695662</td>\n",
       "      <td>26448193</td>\n",
       "      <td>38.018740</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>New York</th>\n",
       "      <td>141297</td>\n",
       "      <td>19651127</td>\n",
       "      <td>139.076746</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Florida</th>\n",
       "      <td>170312</td>\n",
       "      <td>19552860</td>\n",
       "      <td>114.806121</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Illinois</th>\n",
       "      <td>149995</td>\n",
       "      <td>12882135</td>\n",
       "      <td>85.883763</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Area  Population     Density\n",
       "California  423967    38332521   50.000000\n",
       "Texas       695662    26448193   38.018740\n",
       "New York    141297    19651127  139.076746\n",
       "Florida     170312    19552860  114.806121\n",
       "Illinois    149995    12882135   85.883763"
      ]
     },
     "execution_count": 98,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.iloc[3,2] = 79"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Area</th>\n",
       "      <th>Population</th>\n",
       "      <th>Density</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>California</th>\n",
       "      <td>423967</td>\n",
       "      <td>38332521</td>\n",
       "      <td>50.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Texas</th>\n",
       "      <td>695662</td>\n",
       "      <td>26448193</td>\n",
       "      <td>38.018740</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>New York</th>\n",
       "      <td>141297</td>\n",
       "      <td>19651127</td>\n",
       "      <td>139.076746</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Florida</th>\n",
       "      <td>170312</td>\n",
       "      <td>19552860</td>\n",
       "      <td>79.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Illinois</th>\n",
       "      <td>149995</td>\n",
       "      <td>12882135</td>\n",
       "      <td>85.883763</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Area  Population     Density\n",
       "California  423967    38332521   50.000000\n",
       "Texas       695662    26448193   38.018740\n",
       "New York    141297    19651127  139.076746\n",
       "Florida     170312    19552860   79.000000\n",
       "Illinois    149995    12882135   85.883763"
      ]
     },
     "execution_count": 100,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Area</th>\n",
       "      <th>Population</th>\n",
       "      <th>Density</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>California</th>\n",
       "      <td>423967</td>\n",
       "      <td>38332521</td>\n",
       "      <td>50.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Texas</th>\n",
       "      <td>695662</td>\n",
       "      <td>26448193</td>\n",
       "      <td>38.018740</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>New York</th>\n",
       "      <td>141297</td>\n",
       "      <td>19651127</td>\n",
       "      <td>139.076746</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Area  Population     Density\n",
       "California  423967    38332521   50.000000\n",
       "Texas       695662    26448193   38.018740\n",
       "New York    141297    19651127  139.076746"
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.iloc[:3, :3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.DataFrame(np.random.randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>W</th>\n",
       "      <th>X</th>\n",
       "      <th>Y</th>\n",
       "      <th>Z</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>A</th>\n",
       "      <td>1.294733</td>\n",
       "      <td>-0.153451</td>\n",
       "      <td>0.202625</td>\n",
       "      <td>-0.927976</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>B</th>\n",
       "      <td>0.830072</td>\n",
       "      <td>-0.672571</td>\n",
       "      <td>1.070436</td>\n",
       "      <td>1.313183</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>C</th>\n",
       "      <td>0.729096</td>\n",
       "      <td>-0.839462</td>\n",
       "      <td>-0.940769</td>\n",
       "      <td>1.680892</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>D</th>\n",
       "      <td>0.030773</td>\n",
       "      <td>-0.343063</td>\n",
       "      <td>0.070732</td>\n",
       "      <td>-1.758303</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>E</th>\n",
       "      <td>0.005508</td>\n",
       "      <td>-2.095032</td>\n",
       "      <td>1.178295</td>\n",
       "      <td>1.108041</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          W         X         Y         Z\n",
       "A  1.294733 -0.153451  0.202625 -0.927976\n",
       "B  0.830072 -0.672571  1.070436  1.313183\n",
       "C  0.729096 -0.839462 -0.940769  1.680892\n",
       "D  0.030773 -0.343063  0.070732 -1.758303\n",
       "E  0.005508 -2.095032  1.178295  1.108041"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "result = df['W'] > 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>W</th>\n",
       "      <th>X</th>\n",
       "      <th>Y</th>\n",
       "      <th>Z</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>A</th>\n",
       "      <td>1.294733</td>\n",
       "      <td>-0.153451</td>\n",
       "      <td>0.202625</td>\n",
       "      <td>-0.927976</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>B</th>\n",
       "      <td>0.830072</td>\n",
       "      <td>-0.672571</td>\n",
       "      <td>1.070436</td>\n",
       "      <td>1.313183</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>D</th>\n",
       "      <td>0.030773</td>\n",
       "      <td>-0.343063</td>\n",
       "      <td>0.070732</td>\n",
       "      <td>-1.758303</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>E</th>\n",
       "      <td>0.005508</td>\n",
       "      <td>-2.095032</td>\n",
       "      <td>1.178295</td>\n",
       "      <td>1.108041</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          W         X         Y         Z\n",
       "A  1.294733 -0.153451  0.202625 -0.927976\n",
       "B  0.830072 -0.672571  1.070436  1.313183\n",
       "D  0.030773 -0.343063  0.070732 -1.758303\n",
       "E  0.005508 -2.095032  1.178295  1.108041"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df['Y'] > 0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>W</th>\n",
       "      <th>X</th>\n",
       "      <th>Y</th>\n",
       "      <th>Z</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>A</th>\n",
       "      <td>1.294733</td>\n",
       "      <td>-0.153451</td>\n",
       "      <td>0.202625</td>\n",
       "      <td>-0.927976</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>B</th>\n",
       "      <td>0.830072</td>\n",
       "      <td>-0.672571</td>\n",
       "      <td>1.070436</td>\n",
       "      <td>1.313183</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>C</th>\n",
       "      <td>0.729096</td>\n",
       "      <td>-0.839462</td>\n",
       "      <td>-0.940769</td>\n",
       "      <td>1.680892</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>D</th>\n",
       "      <td>0.030773</td>\n",
       "      <td>-0.343063</td>\n",
       "      <td>0.070732</td>\n",
       "      <td>-1.758303</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>E</th>\n",
       "      <td>0.005508</td>\n",
       "      <td>-2.095032</td>\n",
       "      <td>1.178295</td>\n",
       "      <td>1.108041</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          W         X         Y         Z\n",
       "A  1.294733 -0.153451  0.202625 -0.927976\n",
       "B  0.830072 -0.672571  1.070436  1.313183\n",
       "C  0.729096 -0.839462 -0.940769  1.680892\n",
       "D  0.030773 -0.343063  0.070732 -1.758303\n",
       "E  0.005508 -2.095032  1.178295  1.108041"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[(df['W'] > 0) | (df['Z'] >0)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>col1</th>\n",
       "      <th>col2</th>\n",
       "      <th>col3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>444</td>\n",
       "      <td>abc</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>555</td>\n",
       "      <td>def</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>666</td>\n",
       "      <td>ghi</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>444</td>\n",
       "      <td>xyz</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   col1  col2 col3\n",
       "0     1   444  abc\n",
       "1     2   555  def\n",
       "2     3   666  ghi\n",
       "3     4   444  xyz"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>col1</th>\n",
       "      <th>col2</th>\n",
       "      <th>col3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>444</td>\n",
       "      <td>abc</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>555</td>\n",
       "      <td>def</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>666</td>\n",
       "      <td>ghi</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>444</td>\n",
       "      <td>xyz</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   col1  col2 col3\n",
       "0     1   444  abc\n",
       "1     2   555  def\n",
       "2     3   666  ghi\n",
       "3     4   444  xyz"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "555    1\n",
       "2      1\n",
       "def    1\n",
       "Name: 1, dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.loc[1].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>col1</th>\n",
       "      <th>col2</th>\n",
       "      <th>col3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>666</td>\n",
       "      <td>ghi</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>444</td>\n",
       "      <td>xyz</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   col1  col2 col3\n",
       "2     3   666  ghi\n",
       "3     4   444  xyz"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df['col1']>2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def times2(x):\n",
    "    return x*2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0     888\n",
       "1    1110\n",
       "2    1332\n",
       "3     888\n",
       "Name: col2, dtype: int64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['col2'].apply(times2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    2\n",
       "1    4\n",
       "2    6\n",
       "3    8\n",
       "Name: col1, dtype: int64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['col1'].apply(lambda x: x*2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "#DATA INPUT AND OUTPUT WITH PANDAS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'C:\\\\Users\\\\THINKPAD T540p'"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pwd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('example')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.to_csv('My_example', index = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "df1 = pd.read_csv('My_output')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>6</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8</td>\n",
       "      <td>9</td>\n",
       "      <td>10</td>\n",
       "      <td>11</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>12</td>\n",
       "      <td>13</td>\n",
       "      <td>14</td>\n",
       "      <td>15</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    a   b   c   d\n",
       "0   0   1   2   3\n",
       "1   4   5   6   7\n",
       "2   8   9  10  11\n",
       "3  12  13  14  15"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>a</th>\n",
       "      <th>b</th>\n",
       "      <th>c</th>\n",
       "      <th>d</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>6</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>8</td>\n",
       "      <td>9</td>\n",
       "      <td>10</td>\n",
       "      <td>11</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>12</td>\n",
       "      <td>13</td>\n",
       "      <td>14</td>\n",
       "      <td>15</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0   a   b   c   d\n",
       "0           0   0   1   2   3\n",
       "1           1   4   5   6   7\n",
       "2           2   8   9  10  11\n",
       "3           3  12  13  14  15"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.read_excel('Excel_Sample.xlsx', sheet_name = 'Sheet1')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.to_excel('My_sample_sheet.xlsx', sheet_name='NewSheet')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "###Pandas Revision\n",
    "list1 = [2,3,4]\n",
    "labels = ['a','b','c']\n",
    "arr = np.array([10,20,30])\n",
    "d = {'a':10,'b':20,'c':30}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "ser = pd.Series(arr,labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "a    10\n",
       "b    20\n",
       "c    30\n",
       "dtype: int32"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ser"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "ser1 = pd.Series(d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "a    10\n",
       "b    20\n",
       "c    30\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ser1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "##Working with DataFrames"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "outside = ['G1','G1','G1','G2','G2','G2']\n",
    "inside = [1,2,3,1,2,3]\n",
    "hier_index = list(zip(outside,inside))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "hier_index = pd.MultiIndex.from_tuples(hier_index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MultiIndex([('G1', 1),\n",
       "            ('G1', 2),\n",
       "            ('G1', 3),\n",
       "            ('G2', 1),\n",
       "            ('G2', 2),\n",
       "            ('G2', 3)],\n",
       "           )"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "hier_index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.DataFrame(np.random.randn(6,2), index=hier_index, columns = ['A','B'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th>A</th>\n",
       "      <th>B</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">G1</th>\n",
       "      <th>1</th>\n",
       "      <td>0.178269</td>\n",
       "      <td>0.458649</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.962576</td>\n",
       "      <td>-2.354907</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.401862</td>\n",
       "      <td>0.231734</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"3\" valign=\"top\">G2</th>\n",
       "      <th>1</th>\n",
       "      <td>-1.478667</td>\n",
       "      <td>-0.369395</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-0.940614</td>\n",
       "      <td>0.451432</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-0.452433</td>\n",
       "      <td>0.321582</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             A         B\n",
       "G1 1  0.178269  0.458649\n",
       "   2  0.962576 -2.354907\n",
       "   3  0.401862  0.231734\n",
       "G2 1 -1.478667 -0.369395\n",
       "   2 -0.940614  0.451432\n",
       "   3 -0.452433  0.321582"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "state": {},
    "version_major": 2,
    "version_minor": 0
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
