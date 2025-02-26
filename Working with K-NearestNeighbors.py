{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = pd.read_csv('Social_Network_Ads.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
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
       "      <th>User ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Purchased</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>15624510</td>\n",
       "      <td>Male</td>\n",
       "      <td>19</td>\n",
       "      <td>19000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>15810944</td>\n",
       "      <td>Male</td>\n",
       "      <td>35</td>\n",
       "      <td>20000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>15668575</td>\n",
       "      <td>Female</td>\n",
       "      <td>26</td>\n",
       "      <td>43000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>15603246</td>\n",
       "      <td>Female</td>\n",
       "      <td>27</td>\n",
       "      <td>57000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>15804002</td>\n",
       "      <td>Male</td>\n",
       "      <td>19</td>\n",
       "      <td>76000</td>\n",
       "      <td>0</td>\n",
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
       "      <th>395</th>\n",
       "      <td>15691863</td>\n",
       "      <td>Female</td>\n",
       "      <td>46</td>\n",
       "      <td>41000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>396</th>\n",
       "      <td>15706071</td>\n",
       "      <td>Male</td>\n",
       "      <td>51</td>\n",
       "      <td>23000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>397</th>\n",
       "      <td>15654296</td>\n",
       "      <td>Female</td>\n",
       "      <td>50</td>\n",
       "      <td>20000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>398</th>\n",
       "      <td>15755018</td>\n",
       "      <td>Male</td>\n",
       "      <td>36</td>\n",
       "      <td>33000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>399</th>\n",
       "      <td>15594041</td>\n",
       "      <td>Female</td>\n",
       "      <td>49</td>\n",
       "      <td>36000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>400 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      User ID  Gender  Age  EstimatedSalary  Purchased\n",
       "0    15624510    Male   19            19000          0\n",
       "1    15810944    Male   35            20000          0\n",
       "2    15668575  Female   26            43000          0\n",
       "3    15603246  Female   27            57000          0\n",
       "4    15804002    Male   19            76000          0\n",
       "..        ...     ...  ...              ...        ...\n",
       "395  15691863  Female   46            41000          1\n",
       "396  15706071    Male   51            23000          1\n",
       "397  15654296  Female   50            20000          1\n",
       "398  15755018    Male   36            33000          0\n",
       "399  15594041  Female   49            36000          1\n",
       "\n",
       "[400 rows x 5 columns]"
      ]
     },
     "execution_count": 3,
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
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = dataset.iloc[:,2:4].values\n",
    "y = dataset.iloc[:,-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Splitting the data into train and test set\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "##Applying Standard Scaler\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "sc_X = StandardScaler()\n",
    "X_train = sc_X.fit_transform(X_train)\n",
    "X_test = sc_X.transform(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',\n",
       "                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,\n",
       "                     weights='uniform')"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "classifier = KNeighborsClassifier(5, metric = 'minkowski', p=2)\n",
    "classifier.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Making predictions using the test set\n",
    "y_pred = classifier.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[64,  4],\n",
       "       [ 3, 29]], dtype=int64)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##Applying the confusion matrix\n",
    "from sklearn.metrics import confusion_matrix \n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "cm\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.\n",
      "'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
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
    "###Visualizing the Train test Results\n",
    "from matplotlib.colors import ListedColormap\n",
    "X_set, y_set = X_train, y_train\n",
    "X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),\n",
    "                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))\n",
    "plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),\n",
    "             alpha = 0.75, cmap = ListedColormap(('red', 'green')))\n",
    "plt.xlim(X1.min(), X1.max())\n",
    "plt.ylim(X2.min(), X2.max())\n",
    "for i, j in enumerate(np.unique(y_set)):\n",
    "    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],\n",
    "                c = ListedColormap(('red', 'green'))(i), label = j)\n",
    "plt.title('K-NN (Training set)')\n",
    "plt.xlabel('Age')\n",
    "plt.ylabel('Estimated Salary')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.\n",
      "'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
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
    "###Visualizing the Test set Results\n",
    "from matplotlib.colors import ListedColormap\n",
    "X_set, y_set = X_test, y_test\n",
    "X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),\n",
    "                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))\n",
    "plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),\n",
    "             alpha = 0.75, cmap = ListedColormap(('red', 'green')))\n",
    "plt.xlim(X1.min(), X1.max())\n",
    "plt.ylim(X2.min(), X2.max())\n",
    "for i, j in enumerate(np.unique(y_set)):\n",
    "    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],\n",
    "                c = ListedColormap(('red', 'green'))(i), label = j)\n",
    "plt.title('K-NN (Test set)')\n",
    "plt.xlabel('Age')\n",
    "plt.ylabel('Estimated Salary')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    " "
   ]
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
