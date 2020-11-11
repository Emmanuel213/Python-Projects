{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "###Importing the required datasets\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "##Importing the required datasets\n",
    "dataset = pd.read_csv('Mall_Customers.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
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
       "      <th>CustomerID</th>\n",
       "      <th>Genre</th>\n",
       "      <th>Age</th>\n",
       "      <th>Annual Income (k$)</th>\n",
       "      <th>Spending Score (1-100)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Male</td>\n",
       "      <td>19</td>\n",
       "      <td>15</td>\n",
       "      <td>39</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Male</td>\n",
       "      <td>21</td>\n",
       "      <td>15</td>\n",
       "      <td>81</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Female</td>\n",
       "      <td>20</td>\n",
       "      <td>16</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Female</td>\n",
       "      <td>23</td>\n",
       "      <td>16</td>\n",
       "      <td>77</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Female</td>\n",
       "      <td>31</td>\n",
       "      <td>17</td>\n",
       "      <td>40</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>195</th>\n",
       "      <td>196</td>\n",
       "      <td>Female</td>\n",
       "      <td>35</td>\n",
       "      <td>120</td>\n",
       "      <td>79</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>196</th>\n",
       "      <td>197</td>\n",
       "      <td>Female</td>\n",
       "      <td>45</td>\n",
       "      <td>126</td>\n",
       "      <td>28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>197</th>\n",
       "      <td>198</td>\n",
       "      <td>Male</td>\n",
       "      <td>32</td>\n",
       "      <td>126</td>\n",
       "      <td>74</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>198</th>\n",
       "      <td>199</td>\n",
       "      <td>Male</td>\n",
       "      <td>32</td>\n",
       "      <td>137</td>\n",
       "      <td>18</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>199</th>\n",
       "      <td>200</td>\n",
       "      <td>Male</td>\n",
       "      <td>30</td>\n",
       "      <td>137</td>\n",
       "      <td>83</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>200 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     CustomerID   Genre  Age  Annual Income (k$)  Spending Score (1-100)\n",
       "0             1    Male   19                  15                      39\n",
       "1             2    Male   21                  15                      81\n",
       "2             3  Female   20                  16                       6\n",
       "3             4  Female   23                  16                      77\n",
       "4             5  Female   31                  17                      40\n",
       "..          ...     ...  ...                 ...                     ...\n",
       "195         196  Female   35                 120                      79\n",
       "196         197  Female   45                 126                      28\n",
       "197         198    Male   32                 126                      74\n",
       "198         199    Male   32                 137                      18\n",
       "199         200    Male   30                 137                      83\n",
       "\n",
       "[200 rows x 5 columns]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = dataset.iloc[:,[3,4]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
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
       "      <th>Annual Income (k$)</th>\n",
       "      <th>Spending Score (1-100)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>15</td>\n",
       "      <td>39</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>15</td>\n",
       "      <td>81</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>16</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>16</td>\n",
       "      <td>77</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>17</td>\n",
       "      <td>40</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>195</th>\n",
       "      <td>120</td>\n",
       "      <td>79</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>196</th>\n",
       "      <td>126</td>\n",
       "      <td>28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>197</th>\n",
       "      <td>126</td>\n",
       "      <td>74</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>198</th>\n",
       "      <td>137</td>\n",
       "      <td>18</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>199</th>\n",
       "      <td>137</td>\n",
       "      <td>83</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>200 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Annual Income (k$)  Spending Score (1-100)\n",
       "0                    15                      39\n",
       "1                    15                      81\n",
       "2                    16                       6\n",
       "3                    16                      77\n",
       "4                    17                      40\n",
       "..                  ...                     ...\n",
       "195                 120                      79\n",
       "196                 126                      28\n",
       "197                 126                      74\n",
       "198                 137                      18\n",
       "199                 137                      83\n",
       "\n",
       "[200 rows x 2 columns]"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Elbow - Method')"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZcAAAEXCAYAAABh1gnVAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nO3deXxU5dn/8c+VnSyQsO9rqLiyGFFRUbRV3Ipa+1S7QK0VF8SltdW2v9bWPn2qdrNqxaK41a3uu1CryKIIBkQBEdkl7MgWCBAC1++POYEhhiGQmZxJ8n2/XvOaM/dZ5pppzZf7nHvuY+6OiIhIPKWEXYCIiDQ8ChcREYk7hYuIiMSdwkVEROJO4SIiInGncBERkbhTuIgEzOyHZjY56rWbWWGYNdUVM3vXzH4cp2M9Ymb/G49jSf2lcJFGxcyWmNk2M9sS9bg37Lpqwsy6BoE3o0p7SzMrN7MlNTzOb83s8YQUKRJIC7sAkRCc7+7/DbuIWsgxs6PcfXbw+rvAYiAzxJpE9qGei0hs55jZIjNbZ2Z/MrMUADNLMbP/Z2ZLzWyNmT1mZs2CdY+a2U+D5Q5Bb+Oa4HWhma03M6tFTf8ChkW9Hgo8Fr2BmbU3s+fNbK2ZLTaz64L2wcAvge8EvbaPo3brYmbvmVmpmf3HzFpGHe+bZjbHzDYGp9AOj1rX18xmBPv9G8iqxWeTBkLhIhLbhUAR0A8YAvwoaP9h8BgEdAdygcrTaxOA04LlU4FFwTPAQGCS127epceBS8wsNfgjnwdMrVwZBOCrwMdAB+AM4AYzO8vdxwL/B/zb3XPdvXfUcb8LXAa0BjKAm4LjfQ14CrgBaAW8AbxqZhlmlgG8RCTwmgPPAt+qxWeTBkLhIo3RS8G/wCsfV8TY9g53X+/uXwB3AZcG7d8D/urui9x9C/ALIn/w04iEyynBH/mBwJ3AScF+pwbra6MEmAd8nUgP5rEq648DWrn7be5e7u6LgAeASw5w3Ifd/XN33wY8A/QJ2r8DvO7ub7n7TuDPQBNgAHACkA7c5e473f054MNafj5pAHTNRRqjCw7imsuyqOWlQPtguX3wOnpdGtDG3Rea2RYif5xPAX4PXG5mhxEJl7ureyMzmwN0CV6e7e6TYtT1GJGe0wAiAdYzal0XoL2ZbYxqSwViHQ9gVdRyGZHeGFT5rO6+28yWEekV7QKWV+mJRX8v0kgpXERi6wTMCZY7AyuC5RXsDYLKdRXA6uD1BOBiIMPdl5vZBCLXRgqAmdW9kbsfeRB1PU/kNNx0d19qZtHhsgxY7O49q9+Vgz0ltwI4uvJFcL2oE7A8OFYHM7OogOkMLDzI95AGRqfFRGL7mZkVmFkn4Hrg30H7U8CNZtbNzHLZex2jIlg/AbgWmBi8fhcYCUx29121LcrdtwKnA9X9NmUasNnMbjazJsG1maPM7Lhg/Wqga+XghBp4BjjXzM4ws3Tgp8AO4H1gCpFQvc7M0szsIqB/LT6aNBAKF2mMXq3yO5cXY2z7MjCdSG/jdWBM0P4QkYvYE4kMA95OJDwqTSByob0yXCYD2VGva83di939Kz2EILzOJ3JabjGwDngQaBZs8mzw/GXV38zs533mAd8H7gmOdT6R4dzl7l4OXETkFN0GItdnXqjFx5IGwnSzMBERiTf1XEREJO4ULiIiEncKFxERiTuFi4iIxJ1+5xJo2bKld+3aNewyRETqlenTp69z91ZV2xUuga5du1JcXBx2GSIi9YqZVTsjg06LiYhI3ClcREQk7hQuIiISdwoXERGJO4WLiIjEncJFRETiTuEiIiJxp3CppbkrN/PvD78IuwwRkaSicKmlJ6Yu5dcvzWHVpu1hlyIikjQULrV05cAe7HJn9MRFYZciIpI0FC611Kl5Nhf27cCT05aybsuOsMsREUkKCpc4uOa0Huyo2M2YyYvDLkVEJCkoXOKge6tczjumPf+aspRNZTvDLkdEJHQKlzgZMagHW3ZU8Mj7S8IuRUQkdAqXOOnVtinfOKIND723mC07KsIuR0QkVAqXOLp2UCGbtu3k8Q+qvb2BiEijoXCJo96d8jmlZ0senLSI7Tt3hV2OiEhoFC5xNvL0nqzbUs7T0/SrfRFpvBQucda/W3P6d2vOPycuYkeFei8i0jgpXBJg5OmFrNy0nRdmLA+7FBGRUChcEuDkwpb07tiMUe8upGLX7rDLERGpcwkLFzPrZGbjzWyumc0xs+uD9t+a2XIzmxk8zona5xdmtsDM5pnZWVHtg4O2BWZ2S1R7NzObambzzezfZpYRtGcGrxcE67sm6nNWx8y49vSefLG+jFc/WVGXby0ikhQS2XOpAH7q7ocDJwAjzOyIYN3f3L1P8HgDIFh3CXAkMBi4z8xSzSwV+AdwNnAEcGnUce4IjtUT2ABcHrRfDmxw90Lgb8F2deqMXq3p1TaPe99ZwO7dXtdvLyISqoSFi7uvdPcZwXIpMBfoEGOXIcDT7r7D3RcDC4D+wWOBuy9y93LgaWCImRlwOvBcsP+jwAVRx3o0WH4OOCPYvs6kpBgjBhWycO1Wxs5ZVZdvLSISujq55hKcluoLTA2arjWzT8zsITMrCNo6AMuidisJ2vbX3gLY6O4VVdr3OVawflOwfdW6hptZsZkVr127tlafsTrnHN2O7i1zuPedBbir9yIijUfCw8XMcoHngRvcfTMwCugB9AFWAn+p3LSa3f0Q2mMda98G99HuXuTuRa1atYr5OQ5FaopxzaBCPl25mfHz1sT9+CIiySqh4WJm6USC5Ql3fwHA3Ve7+y533w08QOS0F0R6Hp2idu8IrIjRvg7IN7O0Ku37HCtY3wxYH99PVzND+rSnY0ET7lHvRUQakUSOFjNgDDDX3f8a1d4uarMLgdnB8ivAJcFIr25AT2Aa8CHQMxgZlkHkov8rHvlLPR64ONh/GPBy1LGGBcsXA+94SH/Z01NTuOrUHnz0xUamLPwyjBJEROpcInsuJwE/AE6vMuz4TjObZWafAIOAGwHcfQ7wDPApMBYYEfRwKoBrgXFEBgU8E2wLcDPwEzNbQOSaypigfQzQImj/CbBn+HIYLj62I22aZnLPOwvCLENEpM6YTtVEFBUVeXFxccKOP2byYn7/2qc8d9WJFHVtnrD3ERGpS2Y23d2LqrbrF/p15NL+nWiek8G949V7EZGGT+FSR7Iz0rj85G68O28ts5dvCrscEZGEUrjUoaEndqFpVhr36tqLiDRwCpc6lJeVzg8HdGXsnFV8vro07HJERBJG4VLHLjupG9kZqdynay8i0oApXOpYQU4GPzihC698vIIl67aGXY6ISEIoXEJw+SndSE9NYdS7C8MuRUQkIRQuIWidl8Ulx3Xi+RklLN+4LexyRETiTuESkuGn9sAMRk9Q70VEGh6FS0g65DfhW/068tSHy1hTuj3sckRE4krhEqKrTu1Bxa7dPDhpcdiliIjElcIlRF1b5vDN3u15/IOlbNhaHnY5IiJxo3AJ2YhBhZSV7+Lh99R7EZGGQ+ESsp5t8jj7qLY8/P4SNm/fGXY5IiJxoXBJAiMGFVK6vYJ/TVkadikiInGhcEkCR3VoxqDDWjFm8mLKyivCLkdEpNYULkni2tN7sn5rOU9O/SLsUkREak3hkiSO7VLAid1bMHriIrbv3BV2OSIitaJwSSIjTy9kTekOnp1eEnYpIiK1onBJIif2aEG/zvnc/+5Cdu7aHXY5IiKHTOGSRMyMkaf3ZPnGbbz00fKwyxEROWQKlyRz2mGtOLJ9U+57dyG7dnvY5YiIHBKFS5IxM64dVMjidVt5fdbKsMsRETkkCpckdNaRbSlsncs/3lnAbvVeRKQeUrgkoZQUY8SgHsxbXcp/564OuxwRkYOmcElS5x/Tns7Ns7l3/ALc1XsRkfpF4ZKk0lJTuOa0HnxSsolJ89eFXY6IyEFRuCSxi/p1pF2zLO59Z0HYpYiIHBSFSxLLSEvhyoHdmbZkPVMXfRl2OSIiNaZwSXKX9O9My9wM7h2v3ouI1B8JCxcz62Rm481srpnNMbPrg/bmZvaWmc0PnguCdjOzu81sgZl9Ymb9oo41LNh+vpkNi2o/1sxmBfvcbWYW6z3qo6z0VK44pTuT5q9j5rKNYZcjIlIjiey5VAA/dffDgROAEWZ2BHAL8La79wTeDl4DnA30DB7DgVEQCQrgVuB4oD9wa1RYjAq2rdxvcNC+v/eol753QheaNUnXtRcRqTcSFi7uvtLdZwTLpcBcoAMwBHg02OxR4IJgeQjwmEd8AOSbWTvgLOAtd1/v7huAt4DBwbqm7j7FI2N1H6tyrOreo17KzUzjRyd1479zV/Ppis1hlyMickB1cs3FzLoCfYGpQBt3XwmRAAJaB5t1AJZF7VYStMVqL6mmnRjvUW/9cEBXcjPT+Me76r2ISPJLeLiYWS7wPHCDu8f6Z7dV0+aH0H4wtQ03s2IzK167du3B7FrnmmWnM/TELrwxayUL124JuxwRkZgSGi5mlk4kWJ5w9xeC5tXBKS2C5zVBewnQKWr3jsCKA7R3rKY91nvsw91Hu3uRuxe1atXq0D5kHbr85G5kpqVw3/iFYZciIhJTIkeLGTAGmOvuf41a9QpQOeJrGPByVPvQYNTYCcCm4JTWOOBMMysILuSfCYwL1pWa2QnBew2tcqzq3qNea5GbyXf7d+GlmctZtr4s7HJERPYrkT2Xk4AfAKeb2czgcQ5wO/ANM5sPfCN4DfAGsAhYADwAXAPg7uuB3wMfBo/bgjaAq4EHg30WAm8G7ft7j3pv+MDupJpx/wT1XkQkeZkmRYwoKiry4uLisMuokV++OIvnikuY+PNBtG2WFXY5ItKImdl0dy+q2q5f6NdDV5/ag13ujJ64KOxSRESqpXCphzo1z+aCPh14ctpS1m3ZEXY5IiJfoXCpp64Z1IMdFbt5aPLisEsREfkKhUs91aNVLucc3Y7HpixlU9nOsMsREdmHwqUeu3ZQIVt2VPDI+0vCLkVEZB8Kl3rs8HZN+frhbXj4/cVs2VERdjkiInsoXOq5a08vZGPZTp74YGnYpYiI7HHAcDGzHmaWGSyfZmbXmVl+4kuTmujTKZ9TerbkgUmL2L5zV9jliIgANeu5PA/sMrNCItO5dAOeTGhVclCuHVTIui3lPD3ti7BLEREBahYuu929ArgQuMvdbwTaJbYsORjHd29B/27NuXf8Ql17EZGkUJNw2WlmlxKZAPK1oC09cSXJofjlOYezbssO7huv+72ISPhqEi6XAScCf3D3xWbWDXg8sWXJwerTKZ+L+nbgwcmLNWOyiITugOHi7p8CNwOVtyxe7O4NZpbhhuRngw8j1Yw/vjk37FJEpJGryWix84GZwNjgdR8zeyXRhcnBa9esCVed2oM3Zq1i6qIvwy5HRBqxmpwW+y3QH9gI4O4ziYwYkyQ0fGB32jfL4rbXPmXXbt1OQUTCUZNwqXD3TVXa9FcrSTXJSOXms3sxZ8Vmnp9REnY5ItJI1SRcZpvZd4FUM+tpZvcA7ye4LqmFb/ZuT9/O+fxp3DwNTRaRUNQkXEYCRwI7gKeAzcANiSxKasfM+M15R7C2dAej3tXQZBGpezUZLVbm7r9y9+PcvShY3l4Xxcmh69u5gAv7duCBSRqaLCJ1ryajxcab2TtVH3VRnNTOzwcfRorB7WM/C7sUEWlk0mqwzU1Ry1nAtwCdyK8HKocm3/Xf+Qw7cT39uzUPuyQRaSRqclpsetTjPXf/CXB8HdQmcXDlwB60a5bFba/NYbeGJotIHanJabHmUY+WZnYW0LYOapM4aJKRyi1n92L2cg1NFpG6U5PTYtOJ/K7FiJwOWwxcnsiiJL6+2bs9j7y/hDvHzeOco9uRk1mT/9lFRA5dTU6LdXP37sFzT3c/090n10VxEh9mxq/3DE1eGHY5ItII7PefsGZ2Uawd3f2F+JcjidKvcwEX9GnP6EmLuKR/JzoWZIddkog0YLHOj5wfY50DCpd65ueDezF2zipuf/Mz7v1uv7DLEZEGbL/h4u6X1WUhknjt85tw5cAe/P3t+fxwwHqKumposogkRk2mf8HMzjWzn5vZbyofiS5MEuPKU7vTtmkWv3v1Uw1NFpGEqclQ5PuB7xCZY8yAbwNdElyXJEh2Rhq3nN2LWcs38cJHy8MuR0QaqJr0XAa4+1Bgg7v/jsgtjzsltixJpG/2bk+fTvncOfYztmrWZBFJgJqEy7bguczM2gM7qcHNwszsITNbY2azo9p+a2bLzWxm8Dgnat0vzGyBmc0LfqhZ2T44aFtgZrdEtXczs6lmNt/M/m1mGUF7ZvB6QbC+aw0+Y6OSkmL85vwjWFO6g/snaGiyiMRfTcLlNTPLB/4EzACWEJl6/0AeAQZX0/43d+8TPN4AMLMjgEuITO0/GLjPzFLNLBX4B3A2cARwabAtwB3BsXoCG9j7w87LifSyCoG/BdtJFf06FzCkT3tGT1xEyQbNmiwi8bXfcDGzdAB3/727b3T354lca+nl7ge8oO/uE4H1NaxjCPC0u+9w98XAAiK3Vu4PLHD3Re5eDjwNDDEzA04Hngv2fxS4IOpYjwbLzwFnBNtLFTcP7oUZ3DF2XtiliEgDE6vnstzMHjCz0yv/OAd//Kve8vhgXWtmnwSnzQqCtg7AsqhtSoK2/bW3ADa6e0WV9n2OFazfFGz/FWY23MyKzax47dq1tfxY9U/7/CYMH9iDVz9ewfSlNf13gIjIgcUKl8OBYuDXwDIzu8vMajsb8iigB9AHWAn8JWivrmfhh9Ae61hfbXQfHdwArahVq1ax6m6wrjq1O22aZnKbhiaLSBztN1zc/Ut3/6e7DyJyemoxcJeZLTSzPxzKm7n7anff5e67gQeC40Kk5xE9Aq0jsCJG+zog38zSqrTvc6xgfTNqfnqu0cnOSOPmwb34uGQTL2posojESY1+ROnuK4AxRHoepcCPD+XNzKxd1MsLgcqRZK8AlwQjvboBPYFpwIdAz2BkWAaRi/6vuLsD44GLg/2HAS9HHWtYsHwx8E6wvezHBX060LtTPneO09BkEYmPmOFiZllm9m0zewFYCJwB/AJof6ADm9lTwBTgMDMrMbPLgTvNbJaZfQIMAm4EcPc5wDPAp8BYYETQw6kArgXGAXOBZ4JtAW4GfmJmC4hcUxkTtI8BWgTtPwH2DF+W6qWkGL857whWb97BPzU0WUTiwPb3j3ozexL4OjCRyCit19x9ex3WVqeKioq8uLg47DJCdd1THzFuzireuek0OuQ3CbscEakHzGy6uxdVbY/VcxkH9HD3i939uYYcLBJx89m9ALjjzc9CrkRE6rtYF/QfdffSuixGwtUhvwlXDuzOKxqaLCK1VKML+tJ4XHlqj8jQ5NfmamiyiBwyhYvsIyczjZ+f1YuPl23k5Y81NFlEDk2s6V+OM7O2Ua+HmtnLZna3mekuUw3YhX070LtjM+54cx5l5RqaLCIHL1bP5Z9AOYCZDQRuBx4jMp3K6MSXJmGpnDV51ebt3D9hUdjliEg9FCtcUt298qrud4DR7v68u/8aKEx8aRKmY7s05/ze7fnnhIUs37jtwDuIiESJGS5R06ucAbwTtS6tmu2lgbl58GEA3DlWQ5NF5ODECpengAlm9jKRG4ZNAjCzQiKnxqSB61iQzfCB3Xl55gqmL90QdjkiUo/E+p3LH4CfErnp18lR83OlACMTX5okg6tO7UHrvEx+/5pmTRaRmos1WiwbmO7uL7r7VjM7zMxuBI5y9xl1V6KEKSczjZ8P7sXMZRt55eMVB95BRITYp8XGAl1hz6mwKUB3YISZ/THxpUmyuKhvB47p2Izb3/xMQ5NFpEZihUuBu88PlocBT7n7SCL3sz8v4ZVJ0qicNXnV5u38U0OTRaQGYoVL9An204G3AIJ72e9OZFGSfIq6Nue8Y9rxz4kLWaGhySJyALHC5RMz+3NwnaUQ+A+AmeXXSWWSdG45uxe7XUOTReTAYoXLFURuJ9wVONPdy4L2I4A/J7guSUIdC7IZfkp3Xpq5ghlfaGiyiOxfrHDJBV519+vd/eOo9s1ELvZLI3T1aT1olZfJba9+iu4eLSL7Eytc7gFaVtPeAfh7YsqRZBeZNfkwDU0WkZhihcvR7j6haqO7jwOOSVxJkuy+1a8jR3Voyu1vfsa28l1hlyMiSShWuKQf4jpp4CJDk49k5abtjJ6oocki8lWxwmW+mZ1TtdHMzgb0F6WR69+tOece0477Jyxk5SYNTRaRfcUKlxuBu8zsETMbGTweJXK95fq6KU+S2S2De7HLnTvHzgu7FBFJMrEmrvwcOBqYQGQ4ctdg+ZhgnTRynZpnc8Up3Xjxo+V8pKHJIhIl1sSVNxAJl3+5+0+Dx0Puvr3uypNkd/VphZGhya9paLKI7BXrtFhH4G5gjZm9a2b/Z2bnmlnzOqpN6oHczDR+dtZhfPSFhiaLyF6xTovd5O4DgLbAL4H1wI+A2Wb2aR3VJ/XAxf06cmT7ptyhockiEojVc6nUBGgKNAseK4CpiSxK6peUFOPW849kxabtPDBJAwlFBNL2t8LMRgNHAqVEwuR94K/uriu38hX9uzXn3KPbMerdhfxPUSfaNssKuyQRCVGsnktnIBNYBSwHSoCNdVGU1E+3nB0ZmvzDh6exaO2WsMsRkRDFuuYyGDiOvTMg/xT40Mz+Y2a/q4vipH7p1DybMcOKWL15O9+89z1e/2Rl2CWJSEhiXnPxiNnAG8CbwHtAD2rwI0oze8jM1pjZ7Ki25mb2lpnND54LgnYzs7vNbIGZfWJm/aL2GRZsP9/MhkW1H2tms4J97jYzi/UeUjdO6dmK1687ha+1yWXEkzP47StzKK/QveVEGptYv3O5zsyeNrNlwEQitzaeB1wE1GQ48iPA4CpttwBvu3tP4O3gNURundwzeAwHRgU1NAduBY4H+gO3RoXFqGDbyv0GH+A9pI60z2/C08NP5PKTu/HI+0v4n39OYbnuXinSqMTquXQFngP6u3t3d/+Bu9/n7h+7+wH/KeruE4kMX442BHg0WH4UuCCq/bGgp/QBkG9m7YCzgLfcfX0wkOAtYHCwrqm7T/HIL/ceq3Ks6t5D6lBGWgq/Pu8IRn2vHwvXbOHcuycxft6asMsSkToS65rLT9z9OXeP54nzNpXHC55bB+0dgGVR25UEbbHaS6ppj/UeEoKzj27HqyNPpl2zJlz28If8edw8du3WL/lFGrqa/M6lLlg1bX4I7Qf3pmbDzazYzIrXrl17sLtLDXVtmcOL1wzgkuM6ce/4BXz/wamsKdUsQiINWV2Hy+rglBbBc+V5khKgU9R2HYn8WDNWe8dq2mO9x1e4+2h3L3L3olatWh3yh5IDy0pP5fZvHcOfv92bj5Zt4Ny7JzN10ZdhlyUiCVLX4fIKUDniaxjwclT70GDU2AnApuCU1jjgTDMrCC7knwmMC9aVmtkJwSixoVWOVd17SBK4+NiOvDTiJPIy0/jug1MZ9e5Cdus0mUiDk7BwMbOngCnAYWZWYmaXA7cD3zCz+cA3gtcQGeq8CFgAPABcA+Du64HfAx8Gj9uCNoCrgQeDfRYSGSpNjPeQJNGrbVNeGXkyg49qyx1jP+OKx4rZWFYedlkiEkemadIjioqKvLi4OOwyGhV357EpS/nf1z+ldV4Wo77fj2M65oddlogcBDOb7u5FVduT5YK+NEJmxrABXXn2qgEAXDxqCv+askT3hRFpABQuEro+nfJ5beTJnFTYgl+/PIfrnp7J1h0VYZclIrWgcJGkUJCTwZhhx/Gzsw7j9U9W8M17J/P56tKwyxKRQ6RwkaSRkmKMGFTI4z8+nk3bKhhy73u8MKPkwDuKSNJRuEjSGdCjJW9cdzLHdGzGT575mF+88Anbd+oOlyL1icJFklLrplk88ePjufq0Hjw1bRnfGvU+S7/cGnZZIlJDChdJWmmpKdw8uBdjhhVRsmEb590zmbGzV4VdlojUgMJFkt4Zh7fhtZEn061lDlc9Pp3/fe1Tdu7SPWJEkpnCReqFTs2zefaqExl6YhcenLyYS0d/wMpNukeMSLJSuEi9kZmWym1DjuKeS/syd+Vmzr17MpPmazZrkWSkcJF65/ze7Xn52pNpmZvB0Iem8be3Ptc9YkSSjMJF6qXC1rm8NOIkLuzTgb+/PZ8fPjyNL7fsCLssEQkoXKTeys5I4y//05s/XnQ0Uxev59y7J1O8pOqdtUUkDAoXqdfMjEv7d+aFqweQmZ7CJaM/4MFJizT5pUjIFC7SIBzVoRmvjjyZMw5vzf++Ppcr/zWdTWU7wy5LpNFSuEiD0TQrnfu/fyz/79zDeeezNZx51wQmfK7RZCJhULhIg2Jm/PiU7rx4zUk0zUpn2EPT+OWLszSFv0gdU7hIg3R0x8hpsitO6cZT077g7L9PYtpiXewXqSsKF2mwstJT+dW5R/D0FSfgON8ZPYX/e2OuZlgWqQMKF2nwju/egrHXD+TS/p0ZPXER598zmVklm8IuS6RBU7hIo5CTmcb/XXg0j1x2HJu37+TC+97jrv9+rgkwRRJE4SKNymmHteY/N5zKece0467/zuei+95nvm6nLBJ3ChdpdJplp3PXJX2573v9KNlQxrn3TOaBiYs0P5lIHClcpNE65+h2/OfGUxnYsxV/eGMul47+gC++LAu7LJEGQeEijVqrvEweGHosf/52b+au3Mzgv0/kialLNX2MSC0pXKTRMzMuPrYjY28cSN/O+fzqxdn88OEPWbVpe9ilidRbCheRQIf8JvzrR8dz25Ajmbr4S8782wRenrlcvRiRQ6BwEYmSkmIMPbErb14/kMLWuVz/9EyueWKG7hUjcpAULiLV6NYyh2evGsDNg3vx9tw1nHXXRN76dHXYZYnUGwoXkf1ITTGuPq0Hr4w8iVZ5WVzxWDE3Pfsxm7drKn+RA1G4iBxAr7ZNeXnESVw7qJAXZpQw+G8TeW/BurDLEklqoYSLmS0xs1lmNtPMioO25mb2lpnND54LgnYzs7vNbIGZfWJm/aKOMyzYfr6ZDYtqPzY4/oJgX6v7TykNSUZaCjeddRjPXz2ArIxUvvfgVG59eTbbyjUJphM3kFwAAA0cSURBVEh1wuy5DHL3Pu5eFLy+BXjb3XsCbwevAc4GegaP4cAoiIQRcCtwPNAfuLUykIJthkftNzjxH0cag76dC3h95ClcdlJXHp2ylHPunsT0pRvCLksk6STTabEhwKPB8qPABVHtj3nEB0C+mbUDzgLecvf17r4BeAsYHKxr6u5TPDKG9LGoY4nUWpOMVG49/0ievOJ4yit28+373+eOsZ+xo0K9GJFKYYWLA/8xs+lmNjxoa+PuKwGC59ZBewdgWdS+JUFbrPaSatq/wsyGm1mxmRWvXavb4crBGdCjJWNvOIVvH9uJUe8uZMi97/Hpis1hlyWSFMIKl5PcvR+RU14jzGxgjG2ru17ih9D+1Ub30e5e5O5FrVq1OlDNIl+Rl5XOHRcfw5hhRXy5tZwh/5jMP8YvoEJT+UsjF0q4uPuK4HkN8CKRayarg1NaBM9rgs1LgE5Ru3cEVhygvWM17SIJc8bhbfjPDQM588i2/GncPC6+fwoL124JuyyR0NR5uJhZjpnlVS4DZwKzgVeAyhFfw4CXg+VXgKHBqLETgE3BabNxwJlmVhBcyD8TGBesKzWzE4JRYkOjjiWSMAU5Gfzju/2459K+LPlyK+fePYkHJi5i0dotms5fGh2r63mTzKw7kd4KQBrwpLv/wcxaAM8AnYEvgG+7+/ogIO4lMuKrDLjM3SuHL/8I+GVwrD+4+8NBexHwCNAEeBMY6Qf4oEVFRV5cXBy/DyqN2prN27nlhVm881mkA56ZlkKPVrl8rU0uPdvk8bU2eXytTS6dCrJJSdFIeam/zGx61Kjfve2alC9C4SLx5u7MXr6Zuas2M391KZ+v3sL81aWsiJptOSs9hcLWuXytdV4QOrl8rU0eHfKbKHSkXthfuKSFUYxIY2BmHN2xGUd3bLZPe+n2ncxfs4XPVwWBs6aU9xau44WPlu/ZJjsjlZ6tc/cETmVvp32zLPSbYKkPFC4idSwvK51+nQvo17lgn/ZNZTuZvyYSOJ+vLmX+mlImfL6W56bvHVmfm5kW6ekEPZzK8GnbVKEjyUXhIpIkmmWnU9S1OUVdm+/TvmFreaSns7p0z+PtuWt4pnhv6ORlpe25jtOz9d5rOq3yMhU6EgqFi0iSK8jJoH+35vTvtm/ofLllx57Tap8H13TGzl7FU2V7f1vcrEn6ntNqPVtHgqewdS5tmip0JLEULiL1VIvcTE7MzeTEHi32tLk767aUBwMISpkXDCJ47eMVbN5esWe7vMw0CtvkUtgql55t9oaOBhJIvChcRBoQM6NVXiat8jIZUNhyT7u7s3bLDhas2cKCNVuYvzryPH7eWp6NuqZTOXotEjp59AjCp0vzbNJSk2kqQkl2CheRRsDMaJ2XReu8LAb0aLnPuo1l5ZHAqQyeNVuYtng9L83cO7FFeqrRrWXOnh5OYetI6HRrmUNmWmpdfxypBxQuIo1cfnZGtQMJtuyoYGFU6CxYU8qcFZt4Y/ZKKn8el2LQpUVOJGwqQ6d1Hj1a55CdoT8vjZn+1xeRauVmptG7Uz69O+Xv07595y4Wrd3KgrVbWLC6dE/4jP9sDRVR09x0yG8SXM/JDXo7eXRvmUN+droGEzQCChcROShZ6akc0b4pR7Rvuk/7zl27Wfrl1j3XdCpDZ8rCL9lRsXeW6JyMVDoUNKFDfhM6FmTToaAJHYPXHQqa0CpXI9kaAoWLiMRFemoKha3zKGydx+Cj9rbv2u0s37CNz1eXsnR9GSUbyli+YRslG7Yx44uNbNq2c5/jZKal7AmajtWEUOu8LFI1oi3pKVxEJKFSU4zOLbLp3CK72vWl23eyfOO2PYGzd7mMt1ZuZt2W8n22T0sx2uc3qTaAOhY0oW2zLNI1si10ChcRCVVeVjq92qbTq23TatdvK98VCZyN2/bp9SzfuI1J89eypnQH0fPvphi0bZoVBE/2V0KofX4TstI1wi3RFC4iktSaZKTuGf5cnR0Vu1i1aXskcIIeT0nQ+5m2eD2rNm//yv10mjVJpyA7nfzsDAqy0ynIztiznJ8T3bZ3mybpqboWdBAULiJSr2WmpdKlRQ5dWuRUu75i125Wbd7O8g2VvZ9trNuygw1lO9lYVs7aYBqdjWXlbC3ftd/3yUhL2Sd0ogMpuq0gpzKQMmjWJL3RXh9SuIhIg5aWmhJcj6n+mk+08ordbNxWzsaynWzYWr4ngPY+712evyYSSBvLdu4zBLuqpllpFORUH0R5WWnkZKaRm1n5nEpOZho5GXvbMtLq5/UjhYuISCAjLWXPTAY15e6U7qhg49adbCgrZ+O2IIiqhNOGsnK+3BKZDWFj2U627Kg48MGBjNQUcrPSyMlM3Sd0Is+p+4TTnoDKSCM3q+q2aWSnp9bZ3HEKFxGRWjAzmmal0zQrfb8j4qpTXrGbrTsq2LKjgq3lFcHyrr1twaN0z/KuPe0by8op2VDG1srtyyuo6U2FczKqBlIqNw/uRd8q9xeqLYWLiEgIMtJSyEjLoCAno9bHcne27awMn0jglG4PQql8b1hVhte+AbaLlAQMVFC4iIjUc2ZGdkZaZD63vLCriaifV4pERCSpKVxERCTuFC4iIhJ3ChcREYk7hYuIiMSdwkVEROJO4SIiInGncBERkbgzr+mcAQ2cma0FloZdRy21BNaFXUQS0fexl76Lfen72Fdtvo8u7t6qaqPCpQExs2J3Lwq7jmSh72MvfRf70vexr0R8HzotJiIicadwERGRuFO4NCyjwy4gyej72Evfxb70fewr7t+HrrmIiEjcqeciIiJxp3AREZG4U7g0AGbWyczGm9lcM5tjZteHXVPYzCzVzD4ys9fCriVsZpZvZs+Z2WfB/0dODLumsJjZjcF/I7PN7Ckzywq7prpkZg+Z2Rozmx3V1tzM3jKz+cFzXO53rHBpGCqAn7r74cAJwAgzOyLkmsJ2PTA37CKSxN+Bse7eC+hNI/1ezKwDcB1Q5O5HAanAJeFWVeceAQZXabsFeNvdewJvB69rTeHSALj7SnefESyXEvnj0SHcqsJjZh2Bc4EHw64lbGbWFBgIjAFw93J33xhuVaFKA5qYWRqQDawIuZ465e4TgfVVmocAjwbLjwIXxOO9FC4NjJl1BfoCU8OtJFR3AT8HdoddSBLoDqwFHg5OEz5oZjlhFxUGd18O/Bn4AlgJbHL3/4RbVVJo4+4rIfIPVaB1PA6qcGlAzCwXeB64wd03h11PGMzsPGCNu08Pu5YkkQb0A0a5e19gK3E67VHfBNcShgDdgPZAjpl9P9yqGi6FSwNhZulEguUJd38h7HpCdBLwTTNbAjwNnG5mj4dbUqhKgBJ3r+zJPkckbBqjrwOL3X2tu+8EXgAGhFxTMlhtZu0Aguc18TiowqUBMDMjck59rrv/Nex6wuTuv3D3ju7elcjF2nfcvdH+69TdVwHLzOywoOkM4NMQSwrTF8AJZpYd/DdzBo10cEMVrwDDguVhwMvxOGhaPA4ioTsJ+AEwy8xmBm2/dPc3QqxJksdI4AkzywAWAZeFXE8o3H2qmT0HzCAywvIjGtk0MGb2FHAa0NLMSoBbgduBZ8zsciIB/O24vJemfxERkXjTaTEREYk7hYuIiMSdwkVEROJO4SIiInGncBERkbhTuIiISNwpXERqwczczP4S9fomM/vtAfYZGkz5PsfMPjWzm4L2R8zs4kOooauZffegixdJIIWLSO3sAC4ys5Y12djMzgZuAM509yOJTMWyqZY1dAUOKlzMLLWW7ykSk8JFpHYqiPzK+8Yabv8L4CZ3XwHg7tvd/YGqG5nZksrAMrMiM3s3WD7VzGYGj4/MLI/IL6xPCdpuDG6U9icz+9DMPjGzK4N9TwtuKvckkdkccszsdTP7OOhJfae2X4ZIJU3/IlJ7/wA+MbM7a7DtUUBtZmy+CRjh7u8Fs2BvJzLL8U3ufh6AmQ0nMp38cWaWCbxnZpVTy/cHjnL3xWb2LWCFu58b7NesFnWJ7EM9F5FaCm5v8BiRuxwm2nvAX83sOiDf3Suq2eZMYGgwz9xUoAXQM1g3zd0XB8uzgK+b2R1mdoq71/b0nMgeCheR+LgLuBw40I245gDH1uB4Fez973PPfd7d/Xbgx0AT4AMz61XNvgaMdPc+waNb1E2xtkYd6/OgllnAH83sNzWoS6RGFC4iceDu64FniARMLH8E7jSztgBmlhn0Qqpawt4Q+lZlo5n1cPdZ7n4HUAz0AkqBvKh9xwFXB/f4wcy+Vt3dJ82sPVDm7o8TuUNjY73PiySArrmIxM9fgGtjbeDub5hZG+C/wT1FHHiomk1/B4wxs1+y7y2rbzCzQcAuIvdleZPI7ZwrzOxj4BHg70RGkM0I3mMt1d8X/WjgT2a2G9gJXF3DzylyQJpyX0RE4k6nxUREJO50WkwkAczsV3z1jn7PuvsfwqhHpK7ptJiIiMSdTouJiEjcKVxERCTuFC4iIhJ3ChcREYm7/w/b2GGxlrv43wAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "##Using elbow method to make our visualizations\n",
    "from sklearn.cluster import KMeans\n",
    "wcss = []\n",
    "for i in range(1,11):\n",
    "    kmeans = KMeans(n_clusters = i,init = 'k-means++', max_iter = 300, n_init = 10, random_state =0) \n",
    "    kmeans.fit(X)\n",
    "    wcss.append(kmeans.inertia_)\n",
    "plt.plot(range(1,11), wcss)\n",
    "plt.xlabel('N_Clusters')\n",
    "plt.ylabel('WCSS Values')\n",
    "plt.title('Elbow - Method')\n",
    "    "
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
       "array([3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1,\n",
       "       3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 0,\n",
       "       3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 2, 0, 2, 4, 2, 4, 2,\n",
       "       0, 2, 4, 2, 4, 2, 4, 2, 4, 2, 0, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2,\n",
       "       4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2,\n",
       "       4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2,\n",
       "       4, 2])"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##Getting the optimal clusters\n",
    "kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)\n",
    "y_kmeans = kmeans.fit_predict(X)\n",
    "y_kmeans"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "'(array([False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False,  True, False,\n       False,  True,  True,  True,  True,  True,  True,  True,  True,\n        True,  True,  True,  True,  True,  True,  True,  True,  True,\n        True,  True,  True,  True,  True,  True,  True,  True,  True,\n        True,  True,  True,  True,  True,  True,  True,  True,  True,\n        True,  True,  True,  True,  True,  True,  True,  True,  True,\n        True,  True,  True,  True,  True,  True,  True,  True,  True,\n        True,  True,  True,  True,  True,  True,  True,  True,  True,\n        True,  True,  True,  True,  True,  True,  True,  True,  True,\n        True,  True,  True,  True,  True,  True, False, False, False,\n        True, False, False, False, False, False,  True, False, False,\n       False, False, False, False, False, False, False,  True, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False]), 0)' is an invalid key",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-20-80c6c60a2bd2>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mscatter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0my_kmeans\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mX\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0my_kmeans\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0ms\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m100\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mc\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m'red'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlabel\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m'Cluster 1'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mscatter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0my_kmeans\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mX\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0my_kmeans\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0ms\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m100\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mc\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m'blue'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlabel\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m'Cluster 2'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mscatter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0my_kmeans\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m2\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mX\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0my_kmeans\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m2\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0ms\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m100\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mc\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m'green'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlabel\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m'Cluster 3'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mscatter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0my_kmeans\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m3\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mX\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0my_kmeans\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m3\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0ms\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m100\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mc\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m'cyan'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlabel\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m'Cluster 4'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mscatter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0my_kmeans\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m4\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mX\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0my_kmeans\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m4\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0ms\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m100\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mc\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m'magenta'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlabel\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m'Cluster 5'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36m__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   2798\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mnlevels\u001b[0m \u001b[1;33m>\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2799\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_multilevel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2800\u001b[1;33m             \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2801\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mis_integer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mindexer\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2802\u001b[0m                 \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[0mindexer\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[1;34m(self, key, method, tolerance)\u001b[0m\n\u001b[0;32m   2644\u001b[0m                 )\n\u001b[0;32m   2645\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2646\u001b[1;33m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2647\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2648\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_maybe_cast_indexer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: '(array([False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False,  True, False,\n       False,  True,  True,  True,  True,  True,  True,  True,  True,\n        True,  True,  True,  True,  True,  True,  True,  True,  True,\n        True,  True,  True,  True,  True,  True,  True,  True,  True,\n        True,  True,  True,  True,  True,  True,  True,  True,  True,\n        True,  True,  True,  True,  True,  True,  True,  True,  True,\n        True,  True,  True,  True,  True,  True,  True,  True,  True,\n        True,  True,  True,  True,  True,  True,  True,  True,  True,\n        True,  True,  True,  True,  True,  True,  True,  True,  True,\n        True,  True,  True,  True,  True,  True, False, False, False,\n        True, False, False, False, False, False,  True, False, False,\n       False, False, False, False, False, False, False,  True, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False, False, False, False, False, False, False, False,\n       False, False]), 0)' is an invalid key"
     ]
    }
   ],
   "source": [
    "plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')\n",
    "plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')\n",
    "plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')\n",
    "plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')\n",
    "plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')\n",
    "plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')\n",
    "plt.title('Clusters of customers')\n",
    "plt.xlabel('Annual Income (k$)')\n",
    "plt.ylabel('Spending Score (1-100)')\n",
    "plt.legend()\n",
    "plt.show()"
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
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
