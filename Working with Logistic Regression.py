{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
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
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
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
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##Importing the dataset\n",
    "dataset = pd.read_csv('Social_Network_Ads.csv')\n",
    "dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = dataset.iloc[:, 2:4].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = dataset.iloc[:, 4].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(400, 2)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
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
   "execution_count": 9,
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
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n",
       "                   intercept_scaling=1, l1_ratio=None, max_iter=100,\n",
       "                   multi_class='auto', n_jobs=None, penalty='l2',\n",
       "                   random_state=0, solver='lbfgs', tol=0.0001, verbose=0,\n",
       "                   warm_start=False)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##fitting our regression model to out train and test set\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "classifier = LogisticRegression(random_state=0)\n",
    "classifier.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = classifier.predict(X_test)"
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
       "array([0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,\n",
       "       0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,\n",
       "       1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,\n",
       "       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1,\n",
       "       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1], dtype=int64)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,\n",
       "       0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,\n",
       "       1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1,\n",
       "       0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1,\n",
       "       1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1], dtype=int64)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[65,  3],\n",
       "       [ 8, 24]], dtype=int64)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##Evaluating the accuracy of our prediction using confusion matrix\n",
    "from sklearn.metrics import confusion_matrix\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "cm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
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
    "plt.title('Logistic Regression (Training set)')\n",
    "plt.xlabel('Age')\n",
    "plt.ylabel('Estimated Salary')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
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
    "plt.title('Logistic Regression (Test set)')\n",
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
