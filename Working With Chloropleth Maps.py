{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import chart_studio.plotly as py\n",
    "import plotly.graph_objs as go"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from plotly.offline import iplot, plot, download_plotlyjs,init_notebook_mode"
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
       "        <script type=\"text/javascript\">\n",
       "        window.PlotlyConfig = {MathJaxConfig: 'local'};\n",
       "        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: \"STIX-Web\"}});}\n",
       "        if (typeof require !== 'undefined') {\n",
       "        require.undef(\"plotly\");\n",
       "        requirejs.config({\n",
       "            paths: {\n",
       "                'plotly': ['https://cdn.plot.ly/plotly-latest.min']\n",
       "            }\n",
       "        });\n",
       "        require(['plotly'], function(Plotly) {\n",
       "            window._Plotly = Plotly;\n",
       "        });\n",
       "        }\n",
       "        </script>\n",
       "        "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "init_notebook_mode(connected=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = dict(type='choropleth',\n",
    "           locations=['AZ','CA','NY'],\n",
    "           locationmode='USA-states',\n",
    "           colorscale='Portland',\n",
    "           text=['Arizona','California','New York'],\n",
    "            z=[1.0,3.0,4.0],\n",
    "            colorbar={'title':'Colorbar Title'}\n",
    "           )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "layout = dict(geo={'scope':'usa'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "choropleth = go.Figure(data=[data], layout=layout)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "linkText": "Export to plot.ly",
        "plotlyServerURL": "https://plot.ly",
        "showLink": false
       },
       "data": [
        {
         "colorbar": {
          "title": {
           "text": "Colorbar Title"
          }
         },
         "colorscale": [
          [
           0,
           "rgb(12,51,131)"
          ],
          [
           0.25,
           "rgb(10,136,186)"
          ],
          [
           0.5,
           "rgb(242,211,56)"
          ],
          [
           0.75,
           "rgb(242,143,56)"
          ],
          [
           1,
           "rgb(217,30,30)"
          ]
         ],
         "locationmode": "USA-states",
         "locations": [
          "AZ",
          "CA",
          "NY"
         ],
         "text": [
          "Arizona",
          "California",
          "New York"
         ],
         "type": "choropleth",
         "z": [
          1,
          3,
          4
         ]
        }
       ],
       "layout": {
        "geo": {
         "scope": "usa"
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        }
       }
      },
      "text/html": [
       "<div>                            <div id=\"a3c8e505-3753-4d7a-9201-bbbbcfcb3f74\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"a3c8e505-3753-4d7a-9201-bbbbcfcb3f74\")) {                    Plotly.newPlot(                        \"a3c8e505-3753-4d7a-9201-bbbbcfcb3f74\",                        [{\"colorbar\": {\"title\": {\"text\": \"Colorbar Title\"}}, \"colorscale\": [[0.0, \"rgb(12,51,131)\"], [0.25, \"rgb(10,136,186)\"], [0.5, \"rgb(242,211,56)\"], [0.75, \"rgb(242,143,56)\"], [1.0, \"rgb(217,30,30)\"]], \"locationmode\": \"USA-states\", \"locations\": [\"AZ\", \"CA\", \"NY\"], \"text\": [\"Arizona\", \"California\", \"New York\"], \"type\": \"choropleth\", \"z\": [1.0, 3.0, 4.0]}],                        {\"geo\": {\"scope\": \"usa\"}, \"template\": {\"data\": {\"bar\": [{\"error_x\": {\"color\": \"#2a3f5f\"}, \"error_y\": {\"color\": \"#2a3f5f\"}, \"marker\": {\"line\": {\"color\": \"#E5ECF6\", \"width\": 0.5}}, \"type\": \"bar\"}], \"barpolar\": [{\"marker\": {\"line\": {\"color\": \"#E5ECF6\", \"width\": 0.5}}, \"type\": \"barpolar\"}], \"carpet\": [{\"aaxis\": {\"endlinecolor\": \"#2a3f5f\", \"gridcolor\": \"white\", \"linecolor\": \"white\", \"minorgridcolor\": \"white\", \"startlinecolor\": \"#2a3f5f\"}, \"baxis\": {\"endlinecolor\": \"#2a3f5f\", \"gridcolor\": \"white\", \"linecolor\": \"white\", \"minorgridcolor\": \"white\", \"startlinecolor\": \"#2a3f5f\"}, \"type\": \"carpet\"}], \"choropleth\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"type\": \"choropleth\"}], \"contour\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"contour\"}], \"contourcarpet\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"type\": \"contourcarpet\"}], \"heatmap\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"heatmap\"}], \"heatmapgl\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"heatmapgl\"}], \"histogram\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"histogram\"}], \"histogram2d\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"histogram2d\"}], \"histogram2dcontour\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"histogram2dcontour\"}], \"mesh3d\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"type\": \"mesh3d\"}], \"parcoords\": [{\"line\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"parcoords\"}], \"pie\": [{\"automargin\": true, \"type\": \"pie\"}], \"scatter\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatter\"}], \"scatter3d\": [{\"line\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatter3d\"}], \"scattercarpet\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scattercarpet\"}], \"scattergeo\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scattergeo\"}], \"scattergl\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scattergl\"}], \"scattermapbox\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scattermapbox\"}], \"scatterpolar\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatterpolar\"}], \"scatterpolargl\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatterpolargl\"}], \"scatterternary\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatterternary\"}], \"surface\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"surface\"}], \"table\": [{\"cells\": {\"fill\": {\"color\": \"#EBF0F8\"}, \"line\": {\"color\": \"white\"}}, \"header\": {\"fill\": {\"color\": \"#C8D4E3\"}, \"line\": {\"color\": \"white\"}}, \"type\": \"table\"}]}, \"layout\": {\"annotationdefaults\": {\"arrowcolor\": \"#2a3f5f\", \"arrowhead\": 0, \"arrowwidth\": 1}, \"coloraxis\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"colorscale\": {\"diverging\": [[0, \"#8e0152\"], [0.1, \"#c51b7d\"], [0.2, \"#de77ae\"], [0.3, \"#f1b6da\"], [0.4, \"#fde0ef\"], [0.5, \"#f7f7f7\"], [0.6, \"#e6f5d0\"], [0.7, \"#b8e186\"], [0.8, \"#7fbc41\"], [0.9, \"#4d9221\"], [1, \"#276419\"]], \"sequential\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"sequentialminus\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]]}, \"colorway\": [\"#636efa\", \"#EF553B\", \"#00cc96\", \"#ab63fa\", \"#FFA15A\", \"#19d3f3\", \"#FF6692\", \"#B6E880\", \"#FF97FF\", \"#FECB52\"], \"font\": {\"color\": \"#2a3f5f\"}, \"geo\": {\"bgcolor\": \"white\", \"lakecolor\": \"white\", \"landcolor\": \"#E5ECF6\", \"showlakes\": true, \"showland\": true, \"subunitcolor\": \"white\"}, \"hoverlabel\": {\"align\": \"left\"}, \"hovermode\": \"closest\", \"mapbox\": {\"style\": \"light\"}, \"paper_bgcolor\": \"white\", \"plot_bgcolor\": \"#E5ECF6\", \"polar\": {\"angularaxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}, \"bgcolor\": \"#E5ECF6\", \"radialaxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}}, \"scene\": {\"xaxis\": {\"backgroundcolor\": \"#E5ECF6\", \"gridcolor\": \"white\", \"gridwidth\": 2, \"linecolor\": \"white\", \"showbackground\": true, \"ticks\": \"\", \"zerolinecolor\": \"white\"}, \"yaxis\": {\"backgroundcolor\": \"#E5ECF6\", \"gridcolor\": \"white\", \"gridwidth\": 2, \"linecolor\": \"white\", \"showbackground\": true, \"ticks\": \"\", \"zerolinecolor\": \"white\"}, \"zaxis\": {\"backgroundcolor\": \"#E5ECF6\", \"gridcolor\": \"white\", \"gridwidth\": 2, \"linecolor\": \"white\", \"showbackground\": true, \"ticks\": \"\", \"zerolinecolor\": \"white\"}}, \"shapedefaults\": {\"line\": {\"color\": \"#2a3f5f\"}}, \"ternary\": {\"aaxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}, \"baxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}, \"bgcolor\": \"#E5ECF6\", \"caxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}}, \"title\": {\"x\": 0.05}, \"xaxis\": {\"automargin\": true, \"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\", \"title\": {\"standoff\": 15}, \"zerolinecolor\": \"white\", \"zerolinewidth\": 2}, \"yaxis\": {\"automargin\": true, \"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\", \"title\": {\"standoff\": 15}, \"zerolinecolor\": \"white\", \"zerolinewidth\": 2}}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('a3c8e505-3753-4d7a-9201-bbbbcfcb3f74');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "iplot(choropleth)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('2011_US_AGRI_Exports')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
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
       "      <th>code</th>\n",
       "      <th>state</th>\n",
       "      <th>category</th>\n",
       "      <th>total exports</th>\n",
       "      <th>beef</th>\n",
       "      <th>pork</th>\n",
       "      <th>poultry</th>\n",
       "      <th>dairy</th>\n",
       "      <th>fruits fresh</th>\n",
       "      <th>fruits proc</th>\n",
       "      <th>total fruits</th>\n",
       "      <th>veggies fresh</th>\n",
       "      <th>veggies proc</th>\n",
       "      <th>total veggies</th>\n",
       "      <th>corn</th>\n",
       "      <th>wheat</th>\n",
       "      <th>cotton</th>\n",
       "      <th>text</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>AL</td>\n",
       "      <td>Alabama</td>\n",
       "      <td>state</td>\n",
       "      <td>1390.63</td>\n",
       "      <td>34.4</td>\n",
       "      <td>10.6</td>\n",
       "      <td>481.0</td>\n",
       "      <td>4.06</td>\n",
       "      <td>8.0</td>\n",
       "      <td>17.1</td>\n",
       "      <td>25.11</td>\n",
       "      <td>5.5</td>\n",
       "      <td>8.9</td>\n",
       "      <td>14.33</td>\n",
       "      <td>34.9</td>\n",
       "      <td>70.0</td>\n",
       "      <td>317.61</td>\n",
       "      <td>Alabama&lt;br&gt;Beef 34.4 Dairy 4.06&lt;br&gt;Fruits 25.1...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>AK</td>\n",
       "      <td>Alaska</td>\n",
       "      <td>state</td>\n",
       "      <td>13.31</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.19</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.6</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.56</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>Alaska&lt;br&gt;Beef 0.2 Dairy 0.19&lt;br&gt;Fruits 0.0 Ve...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>AZ</td>\n",
       "      <td>Arizona</td>\n",
       "      <td>state</td>\n",
       "      <td>1463.17</td>\n",
       "      <td>71.3</td>\n",
       "      <td>17.9</td>\n",
       "      <td>0.0</td>\n",
       "      <td>105.48</td>\n",
       "      <td>19.3</td>\n",
       "      <td>41.0</td>\n",
       "      <td>60.27</td>\n",
       "      <td>147.5</td>\n",
       "      <td>239.4</td>\n",
       "      <td>386.91</td>\n",
       "      <td>7.3</td>\n",
       "      <td>48.7</td>\n",
       "      <td>423.95</td>\n",
       "      <td>Arizona&lt;br&gt;Beef 71.3 Dairy 105.48&lt;br&gt;Fruits 60...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>AR</td>\n",
       "      <td>Arkansas</td>\n",
       "      <td>state</td>\n",
       "      <td>3586.02</td>\n",
       "      <td>53.2</td>\n",
       "      <td>29.4</td>\n",
       "      <td>562.9</td>\n",
       "      <td>3.53</td>\n",
       "      <td>2.2</td>\n",
       "      <td>4.7</td>\n",
       "      <td>6.88</td>\n",
       "      <td>4.4</td>\n",
       "      <td>7.1</td>\n",
       "      <td>11.45</td>\n",
       "      <td>69.5</td>\n",
       "      <td>114.5</td>\n",
       "      <td>665.44</td>\n",
       "      <td>Arkansas&lt;br&gt;Beef 53.2 Dairy 3.53&lt;br&gt;Fruits 6.8...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>CA</td>\n",
       "      <td>California</td>\n",
       "      <td>state</td>\n",
       "      <td>16472.88</td>\n",
       "      <td>228.7</td>\n",
       "      <td>11.1</td>\n",
       "      <td>225.4</td>\n",
       "      <td>929.95</td>\n",
       "      <td>2791.8</td>\n",
       "      <td>5944.6</td>\n",
       "      <td>8736.40</td>\n",
       "      <td>803.2</td>\n",
       "      <td>1303.5</td>\n",
       "      <td>2106.79</td>\n",
       "      <td>34.6</td>\n",
       "      <td>249.3</td>\n",
       "      <td>1064.95</td>\n",
       "      <td>California&lt;br&gt;Beef 228.7 Dairy 929.95&lt;br&gt;Frui...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  code        state category  total exports   beef  pork  poultry   dairy  \\\n",
       "0   AL      Alabama    state        1390.63   34.4  10.6    481.0    4.06   \n",
       "1   AK       Alaska    state          13.31    0.2   0.1      0.0    0.19   \n",
       "2   AZ      Arizona    state        1463.17   71.3  17.9      0.0  105.48   \n",
       "3   AR     Arkansas    state        3586.02   53.2  29.4    562.9    3.53   \n",
       "4   CA   California    state       16472.88  228.7  11.1    225.4  929.95   \n",
       "\n",
       "   fruits fresh  fruits proc  total fruits  veggies fresh  veggies proc  \\\n",
       "0           8.0         17.1         25.11            5.5           8.9   \n",
       "1           0.0          0.0          0.00            0.6           1.0   \n",
       "2          19.3         41.0         60.27          147.5         239.4   \n",
       "3           2.2          4.7          6.88            4.4           7.1   \n",
       "4        2791.8       5944.6       8736.40          803.2        1303.5   \n",
       "\n",
       "   total veggies  corn  wheat   cotton  \\\n",
       "0          14.33  34.9   70.0   317.61   \n",
       "1           1.56   0.0    0.0     0.00   \n",
       "2         386.91   7.3   48.7   423.95   \n",
       "3          11.45  69.5  114.5   665.44   \n",
       "4        2106.79  34.6  249.3  1064.95   \n",
       "\n",
       "                                                text  \n",
       "0  Alabama<br>Beef 34.4 Dairy 4.06<br>Fruits 25.1...  \n",
       "1  Alaska<br>Beef 0.2 Dairy 0.19<br>Fruits 0.0 Ve...  \n",
       "2  Arizona<br>Beef 71.3 Dairy 105.48<br>Fruits 60...  \n",
       "3  Arkansas<br>Beef 53.2 Dairy 3.53<br>Fruits 6.8...  \n",
       "4   California<br>Beef 228.7 Dairy 929.95<br>Frui...  "
      ]
     },
     "execution_count": 10,
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
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = dict(type ='choropleth',\n",
    "           locations= df['code'],\n",
    "           locationmode='USA-states',\n",
    "           colorscale='Earth',\n",
    "            marker = dict(line = dict(color = 'rgb(255,255,255)', width=2)),\n",
    "           z= df['total exports'],\n",
    "           text=df['text'],\n",
    "           colorbar={'title':'Millions USD'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "layout = dict(title = '2011 USA AGRICULTURAL EXPORTS',\n",
    "             geo = dict(scope = 'usa',\n",
    "                       showlakes= True,\n",
    "                        lakecolor = 'rgb(85,173,240)'\n",
    "                       ))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "choromap = go.Figure(data =[data], layout = layout)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "linkText": "Export to plot.ly",
        "plotlyServerURL": "https://plot.ly",
        "showLink": false
       },
       "data": [
        {
         "colorbar": {
          "title": {
           "text": "Millions USD"
          }
         },
         "colorscale": [
          [
           0,
           "rgb(161, 105, 40)"
          ],
          [
           0.16666666666666666,
           "rgb(189, 146, 90)"
          ],
          [
           0.3333333333333333,
           "rgb(214, 189, 141)"
          ],
          [
           0.5,
           "rgb(237, 234, 194)"
          ],
          [
           0.6666666666666666,
           "rgb(181, 200, 184)"
          ],
          [
           0.8333333333333334,
           "rgb(121, 167, 172)"
          ],
          [
           1,
           "rgb(40, 135, 161)"
          ]
         ],
         "locationmode": "USA-states",
         "locations": [
          "AL",
          "AK",
          "AZ",
          "AR",
          "CA",
          "CO",
          "CT",
          "DE",
          "FL",
          "GA",
          "HI",
          "ID",
          "IL",
          "IN",
          "IA",
          "KS",
          "KY",
          "LA",
          "ME",
          "MD",
          "MA",
          "MI",
          "MN",
          "MS",
          "MO",
          "MT",
          "NE",
          "NV",
          "NH",
          "NJ",
          "NM",
          "NY",
          "NC",
          "ND",
          "OH",
          "OK",
          "OR",
          "PA",
          "RI",
          "SC",
          "SD",
          "TN",
          "TX",
          "UT",
          "VT",
          "VA",
          "WA",
          "WV",
          "WI",
          "WY"
         ],
         "marker": {
          "line": {
           "color": "rgb(255,255,255)",
           "width": 2
          }
         },
         "text": [
          "Alabama<br>Beef 34.4 Dairy 4.06<br>Fruits 25.11 Veggies 14.33<br>Wheat 70.0 Corn 34.9",
          "Alaska<br>Beef 0.2 Dairy 0.19<br>Fruits 0.0 Veggies 1.56<br>Wheat 0.0 Corn 0.0",
          "Arizona<br>Beef 71.3 Dairy 105.48<br>Fruits 60.27 Veggies 386.91<br>Wheat 48.7 Corn 7.3",
          "Arkansas<br>Beef 53.2 Dairy 3.53<br>Fruits 6.88 Veggies 11.45<br>Wheat 114.5 Corn 69.5",
          " California<br>Beef 228.7 Dairy 929.95<br>Fruits 8736.4 Veggies 2106.79<br>Wheat 249.3 Corn 34.6",
          "Colorado<br>Beef 261.4 Dairy 71.94<br>Fruits 17.99 Veggies 118.27<br>Wheat 400.5 Corn 183.2",
          "Connecticut<br>Beef 1.1 Dairy 9.49<br>Fruits 13.1 Veggies 11.16<br>Wheat 0.0 Corn 0.0",
          "Delaware<br>Beef 0.4 Dairy 2.3<br>Fruits 1.53 Veggies 20.03<br>Wheat 22.9 Corn 26.9",
          "Florida<br>Beef 42.6 Dairy 66.31<br>Fruits 1371.36 Veggies 450.86<br>Wheat 1.8 Corn 3.5",
          "Georgia<br>Beef 31.0 Dairy 38.38<br>Fruits 233.51 Veggies 154.77<br>Wheat 65.4 Corn 57.8",
          "Hawaii<br>Beef 4.0 Dairy 1.16<br>Fruits 55.51 Veggies 24.83<br>Wheat 0.0 Corn 0.0",
          "Idaho<br>Beef 119.8 Dairy 294.6<br>Fruits 21.64 Veggies 319.19<br>Wheat 568.2 Corn 24.0",
          "Illinois<br>Beef 53.7 Dairy 45.82<br>Fruits 12.53 Veggies 39.95<br>Wheat 223.8 Corn 2228.5",
          "Indiana<br>Beef 21.9 Dairy 89.7<br>Fruits 12.98 Veggies 37.89<br>Wheat 114.0 Corn 1123.2",
          "Iowa<br>Beef 289.8 Dairy 107.0<br>Fruits 3.24 Veggies 7.1<br>Wheat 3.1 Corn 2529.8",
          "Kansas<br>Beef 659.3 Dairy 65.45<br>Fruits 3.11 Veggies 9.32<br>Wheat 1426.5 Corn 457.3",
          "Kentucky<br>Beef 54.8 Dairy 28.27<br>Fruits 6.6 Veggies 0.0<br>Wheat 149.3 Corn 179.1",
          "Louisiana<br>Beef 19.8 Dairy 6.02<br>Fruits 17.83 Veggies 17.25<br>Wheat 78.7 Corn 91.4",
          "Maine<br>Beef 1.4 Dairy 16.18<br>Fruits 52.01 Veggies 62.9<br>Wheat 0.0 Corn 0.0",
          "Maryland<br>Beef 5.6 Dairy 24.81<br>Fruits 12.9 Veggies 20.43<br>Wheat 55.8 Corn 54.1",
          "Massachusetts<br>Beef 0.6 Dairy 5.81<br>Fruits 80.83 Veggies 21.13<br>Wheat 0.0 Corn 0.0",
          "Michigan<br>Beef 37.7 Dairy 214.82<br>Fruits 257.69 Veggies 189.96<br>Wheat 247.0 Corn 381.5",
          "Minnesota<br>Beef 112.3 Dairy 218.05<br>Fruits 7.91 Veggies 120.37<br>Wheat 538.1 Corn 1264.3",
          "Mississippi<br>Beef 12.8 Dairy 5.45<br>Fruits 17.04 Veggies 27.87<br>Wheat 102.2 Corn 110.0",
          "Missouri<br>Beef 137.2 Dairy 34.26<br>Fruits 13.18 Veggies 17.9<br>Wheat 161.7 Corn 428.8",
          "Montana<br>Beef 105.0 Dairy 6.82<br>Fruits 3.3 Veggies 45.27<br>Wheat 1198.1 Corn 5.4",
          "Nebraska<br>Beef 762.2 Dairy 30.07<br>Fruits 2.16 Veggies 53.5<br>Wheat 292.3 Corn 1735.9",
          "Nevada<br>Beef 21.8 Dairy 16.57<br>Fruits 1.19 Veggies 27.93<br>Wheat 5.4 Corn 0.0",
          "New Hampshire<br>Beef 0.6 Dairy 7.46<br>Fruits 7.98 Veggies 4.5<br>Wheat 0.0 Corn 0.0",
          "New Jersey<br>Beef 0.8 Dairy 3.37<br>Fruits 109.45 Veggies 56.54<br>Wheat 6.7 Corn 10.1",
          "New Mexico<br>Beef 117.2 Dairy 191.01<br>Fruits 101.9 Veggies 43.88<br>Wheat 13.9 Corn 11.2",
          "New York<br>Beef 22.2 Dairy 331.8<br>Fruits 202.56 Veggies 143.37<br>Wheat 29.9 Corn 106.1",
          "North Carolina<br>Beef 24.8 Dairy 24.9<br>Fruits 74.47 Veggies 150.45<br>Wheat 200.3 Corn 92.2",
          "North Dakota<br>Beef 78.5 Dairy 8.14<br>Fruits 0.25 Veggies 130.79<br>Wheat 1664.5 Corn 236.1",
          "Ohio<br>Beef 36.2 Dairy 134.57<br>Fruits 27.21 Veggies 53.53<br>Wheat 207.4 Corn 535.1",
          "Oklahoma<br>Beef 337.6 Dairy 24.35<br>Fruits 9.24 Veggies 8.9<br>Wheat 324.8 Corn 27.5",
          "Oregon<br>Beef 58.8 Dairy 63.66<br>Fruits 315.04 Veggies 126.5<br>Wheat 320.3 Corn 11.7",
          "Pennsylvania<br>Beef 50.9 Dairy 280.87<br>Fruits 89.48 Veggies 38.26<br>Wheat 41.0 Corn 112.1",
          "Rhode Island<br>Beef 0.1 Dairy 0.52<br>Fruits 2.83 Veggies 3.02<br>Wheat 0.0 Corn 0.0",
          "South Carolina<br>Beef 15.2 Dairy 7.62<br>Fruits 53.45 Veggies 42.66<br>Wheat 55.3 Corn 32.1",
          "South Dakota<br>Beef 193.5 Dairy 46.77<br>Fruits 0.8 Veggies 4.06<br>Wheat 704.5 Corn 643.6",
          "Tennessee<br>Beef 51.1 Dairy 21.18<br>Fruits 6.23 Veggies 24.67<br>Wheat 100.0 Corn 88.8",
          "Texas<br>Beef 961.0 Dairy 240.55<br>Fruits 99.9 Veggies 115.23<br>Wheat 309.7 Corn 167.2",
          "Utah<br>Beef 27.9 Dairy 48.6<br>Fruits 12.34 Veggies 6.6<br>Wheat 42.8 Corn 5.3",
          "Vermont<br>Beef 6.2 Dairy 65.98<br>Fruits 8.01 Veggies 4.05<br>Wheat 0.0 Corn 0.0",
          "Virginia<br>Beef 39.5 Dairy 47.85<br>Fruits 36.48 Veggies 27.25<br>Wheat 77.5 Corn 39.5",
          "Washington<br>Beef 59.2 Dairy 154.18<br>Fruits 1738.57 Veggies 363.79<br>Wheat 786.3 Corn 29.5",
          "West Virginia<br>Beef 12.0 Dairy 3.9<br>Fruits 11.54 Veggies 0.0<br>Wheat 1.6 Corn 3.5",
          "Wisconsin<br>Beef 107.3 Dairy 633.6<br>Fruits 133.8 Veggies 148.99<br>Wheat 96.7 Corn 460.5",
          "Wyoming<br>Beef 75.1 Dairy 2.89<br>Fruits 0.17 Veggies 10.23<br>Wheat 20.7 Corn 9.0"
         ],
         "type": "choropleth",
         "z": [
          1390.63,
          13.31,
          1463.17,
          3586.02,
          16472.88,
          1851.33,
          259.62,
          282.19,
          3764.09,
          2860.84,
          401.84,
          2078.89,
          8709.48,
          5050.23,
          11273.76,
          4589.01,
          1889.15,
          1914.23,
          278.37,
          692.75,
          248.65,
          3164.16,
          7192.33,
          2170.8,
          3933.42,
          1718,
          7114.13,
          139.89,
          73.06,
          500.4,
          751.58,
          1488.9,
          3806.05,
          3761.96,
          3979.79,
          1646.41,
          1794.57,
          1969.87,
          31.59,
          929.93,
          3770.19,
          1535.13,
          6648.22,
          453.39,
          180.14,
          1146.48,
          3894.81,
          138.89,
          3090.23,
          349.69
         ]
        }
       ],
       "layout": {
        "geo": {
         "lakecolor": "rgb(85,173,240)",
         "scope": "usa",
         "showlakes": true
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "2011 USA AGRICULTURAL EXPORTS"
        }
       }
      },
      "text/html": [
       "<div>                            <div id=\"217aa6f5-7ba7-4f41-b1e4-9a7c773c7dfc\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"217aa6f5-7ba7-4f41-b1e4-9a7c773c7dfc\")) {                    Plotly.newPlot(                        \"217aa6f5-7ba7-4f41-b1e4-9a7c773c7dfc\",                        [{\"colorbar\": {\"title\": {\"text\": \"Millions USD\"}}, \"colorscale\": [[0.0, \"rgb(161, 105, 40)\"], [0.16666666666666666, \"rgb(189, 146, 90)\"], [0.3333333333333333, \"rgb(214, 189, 141)\"], [0.5, \"rgb(237, 234, 194)\"], [0.6666666666666666, \"rgb(181, 200, 184)\"], [0.8333333333333334, \"rgb(121, 167, 172)\"], [1.0, \"rgb(40, 135, 161)\"]], \"locationmode\": \"USA-states\", \"locations\": [\"AL\", \"AK\", \"AZ\", \"AR\", \"CA\", \"CO\", \"CT\", \"DE\", \"FL\", \"GA\", \"HI\", \"ID\", \"IL\", \"IN\", \"IA\", \"KS\", \"KY\", \"LA\", \"ME\", \"MD\", \"MA\", \"MI\", \"MN\", \"MS\", \"MO\", \"MT\", \"NE\", \"NV\", \"NH\", \"NJ\", \"NM\", \"NY\", \"NC\", \"ND\", \"OH\", \"OK\", \"OR\", \"PA\", \"RI\", \"SC\", \"SD\", \"TN\", \"TX\", \"UT\", \"VT\", \"VA\", \"WA\", \"WV\", \"WI\", \"WY\"], \"marker\": {\"line\": {\"color\": \"rgb(255,255,255)\", \"width\": 2}}, \"text\": [\"Alabama<br>Beef 34.4 Dairy 4.06<br>Fruits 25.11 Veggies 14.33<br>Wheat 70.0 Corn 34.9\", \"Alaska<br>Beef 0.2 Dairy 0.19<br>Fruits 0.0 Veggies 1.56<br>Wheat 0.0 Corn 0.0\", \"Arizona<br>Beef 71.3 Dairy 105.48<br>Fruits 60.27 Veggies 386.91<br>Wheat 48.7 Corn 7.3\", \"Arkansas<br>Beef 53.2 Dairy 3.53<br>Fruits 6.88 Veggies 11.45<br>Wheat 114.5 Corn 69.5\", \" California<br>Beef 228.7 Dairy 929.95<br>Fruits 8736.4 Veggies 2106.79<br>Wheat 249.3 Corn 34.6\", \"Colorado<br>Beef 261.4 Dairy 71.94<br>Fruits 17.99 Veggies 118.27<br>Wheat 400.5 Corn 183.2\", \"Connecticut<br>Beef 1.1 Dairy 9.49<br>Fruits 13.1 Veggies 11.16<br>Wheat 0.0 Corn 0.0\", \"Delaware<br>Beef 0.4 Dairy 2.3<br>Fruits 1.53 Veggies 20.03<br>Wheat 22.9 Corn 26.9\", \"Florida<br>Beef 42.6 Dairy 66.31<br>Fruits 1371.36 Veggies 450.86<br>Wheat 1.8 Corn 3.5\", \"Georgia<br>Beef 31.0 Dairy 38.38<br>Fruits 233.51 Veggies 154.77<br>Wheat 65.4 Corn 57.8\", \"Hawaii<br>Beef 4.0 Dairy 1.16<br>Fruits 55.51 Veggies 24.83<br>Wheat 0.0 Corn 0.0\", \"Idaho<br>Beef 119.8 Dairy 294.6<br>Fruits 21.64 Veggies 319.19<br>Wheat 568.2 Corn 24.0\", \"Illinois<br>Beef 53.7 Dairy 45.82<br>Fruits 12.53 Veggies 39.95<br>Wheat 223.8 Corn 2228.5\", \"Indiana<br>Beef 21.9 Dairy 89.7<br>Fruits 12.98 Veggies 37.89<br>Wheat 114.0 Corn 1123.2\", \"Iowa<br>Beef 289.8 Dairy 107.0<br>Fruits 3.24 Veggies 7.1<br>Wheat 3.1 Corn 2529.8\", \"Kansas<br>Beef 659.3 Dairy 65.45<br>Fruits 3.11 Veggies 9.32<br>Wheat 1426.5 Corn 457.3\", \"Kentucky<br>Beef 54.8 Dairy 28.27<br>Fruits 6.6 Veggies 0.0<br>Wheat 149.3 Corn 179.1\", \"Louisiana<br>Beef 19.8 Dairy 6.02<br>Fruits 17.83 Veggies 17.25<br>Wheat 78.7 Corn 91.4\", \"Maine<br>Beef 1.4 Dairy 16.18<br>Fruits 52.01 Veggies 62.9<br>Wheat 0.0 Corn 0.0\", \"Maryland<br>Beef 5.6 Dairy 24.81<br>Fruits 12.9 Veggies 20.43<br>Wheat 55.8 Corn 54.1\", \"Massachusetts<br>Beef 0.6 Dairy 5.81<br>Fruits 80.83 Veggies 21.13<br>Wheat 0.0 Corn 0.0\", \"Michigan<br>Beef 37.7 Dairy 214.82<br>Fruits 257.69 Veggies 189.96<br>Wheat 247.0 Corn 381.5\", \"Minnesota<br>Beef 112.3 Dairy 218.05<br>Fruits 7.91 Veggies 120.37<br>Wheat 538.1 Corn 1264.3\", \"Mississippi<br>Beef 12.8 Dairy 5.45<br>Fruits 17.04 Veggies 27.87<br>Wheat 102.2 Corn 110.0\", \"Missouri<br>Beef 137.2 Dairy 34.26<br>Fruits 13.18 Veggies 17.9<br>Wheat 161.7 Corn 428.8\", \"Montana<br>Beef 105.0 Dairy 6.82<br>Fruits 3.3 Veggies 45.27<br>Wheat 1198.1 Corn 5.4\", \"Nebraska<br>Beef 762.2 Dairy 30.07<br>Fruits 2.16 Veggies 53.5<br>Wheat 292.3 Corn 1735.9\", \"Nevada<br>Beef 21.8 Dairy 16.57<br>Fruits 1.19 Veggies 27.93<br>Wheat 5.4 Corn 0.0\", \"New Hampshire<br>Beef 0.6 Dairy 7.46<br>Fruits 7.98 Veggies 4.5<br>Wheat 0.0 Corn 0.0\", \"New Jersey<br>Beef 0.8 Dairy 3.37<br>Fruits 109.45 Veggies 56.54<br>Wheat 6.7 Corn 10.1\", \"New Mexico<br>Beef 117.2 Dairy 191.01<br>Fruits 101.9 Veggies 43.88<br>Wheat 13.9 Corn 11.2\", \"New York<br>Beef 22.2 Dairy 331.8<br>Fruits 202.56 Veggies 143.37<br>Wheat 29.9 Corn 106.1\", \"North Carolina<br>Beef 24.8 Dairy 24.9<br>Fruits 74.47 Veggies 150.45<br>Wheat 200.3 Corn 92.2\", \"North Dakota<br>Beef 78.5 Dairy 8.14<br>Fruits 0.25 Veggies 130.79<br>Wheat 1664.5 Corn 236.1\", \"Ohio<br>Beef 36.2 Dairy 134.57<br>Fruits 27.21 Veggies 53.53<br>Wheat 207.4 Corn 535.1\", \"Oklahoma<br>Beef 337.6 Dairy 24.35<br>Fruits 9.24 Veggies 8.9<br>Wheat 324.8 Corn 27.5\", \"Oregon<br>Beef 58.8 Dairy 63.66<br>Fruits 315.04 Veggies 126.5<br>Wheat 320.3 Corn 11.7\", \"Pennsylvania<br>Beef 50.9 Dairy 280.87<br>Fruits 89.48 Veggies 38.26<br>Wheat 41.0 Corn 112.1\", \"Rhode Island<br>Beef 0.1 Dairy 0.52<br>Fruits 2.83 Veggies 3.02<br>Wheat 0.0 Corn 0.0\", \"South Carolina<br>Beef 15.2 Dairy 7.62<br>Fruits 53.45 Veggies 42.66<br>Wheat 55.3 Corn 32.1\", \"South Dakota<br>Beef 193.5 Dairy 46.77<br>Fruits 0.8 Veggies 4.06<br>Wheat 704.5 Corn 643.6\", \"Tennessee<br>Beef 51.1 Dairy 21.18<br>Fruits 6.23 Veggies 24.67<br>Wheat 100.0 Corn 88.8\", \"Texas<br>Beef 961.0 Dairy 240.55<br>Fruits 99.9 Veggies 115.23<br>Wheat 309.7 Corn 167.2\", \"Utah<br>Beef 27.9 Dairy 48.6<br>Fruits 12.34 Veggies 6.6<br>Wheat 42.8 Corn 5.3\", \"Vermont<br>Beef 6.2 Dairy 65.98<br>Fruits 8.01 Veggies 4.05<br>Wheat 0.0 Corn 0.0\", \"Virginia<br>Beef 39.5 Dairy 47.85<br>Fruits 36.48 Veggies 27.25<br>Wheat 77.5 Corn 39.5\", \"Washington<br>Beef 59.2 Dairy 154.18<br>Fruits 1738.57 Veggies 363.79<br>Wheat 786.3 Corn 29.5\", \"West Virginia<br>Beef 12.0 Dairy 3.9<br>Fruits 11.54 Veggies 0.0<br>Wheat 1.6 Corn 3.5\", \"Wisconsin<br>Beef 107.3 Dairy 633.6<br>Fruits 133.8 Veggies 148.99<br>Wheat 96.7 Corn 460.5\", \"Wyoming<br>Beef 75.1 Dairy 2.89<br>Fruits 0.17 Veggies 10.23<br>Wheat 20.7 Corn 9.0\"], \"type\": \"choropleth\", \"z\": [1390.63, 13.31, 1463.17, 3586.02, 16472.88, 1851.33, 259.62, 282.19, 3764.09, 2860.84, 401.84, 2078.89, 8709.48, 5050.23, 11273.76, 4589.01, 1889.15, 1914.23, 278.37, 692.75, 248.65, 3164.16, 7192.33, 2170.8, 3933.42, 1718.0, 7114.13, 139.89, 73.06, 500.4, 751.58, 1488.9, 3806.05, 3761.96, 3979.79, 1646.41, 1794.57, 1969.87, 31.59, 929.93, 3770.19, 1535.13, 6648.22, 453.39, 180.14, 1146.48, 3894.81, 138.89, 3090.23, 349.69]}],                        {\"geo\": {\"lakecolor\": \"rgb(85,173,240)\", \"scope\": \"usa\", \"showlakes\": true}, \"template\": {\"data\": {\"bar\": [{\"error_x\": {\"color\": \"#2a3f5f\"}, \"error_y\": {\"color\": \"#2a3f5f\"}, \"marker\": {\"line\": {\"color\": \"#E5ECF6\", \"width\": 0.5}}, \"type\": \"bar\"}], \"barpolar\": [{\"marker\": {\"line\": {\"color\": \"#E5ECF6\", \"width\": 0.5}}, \"type\": \"barpolar\"}], \"carpet\": [{\"aaxis\": {\"endlinecolor\": \"#2a3f5f\", \"gridcolor\": \"white\", \"linecolor\": \"white\", \"minorgridcolor\": \"white\", \"startlinecolor\": \"#2a3f5f\"}, \"baxis\": {\"endlinecolor\": \"#2a3f5f\", \"gridcolor\": \"white\", \"linecolor\": \"white\", \"minorgridcolor\": \"white\", \"startlinecolor\": \"#2a3f5f\"}, \"type\": \"carpet\"}], \"choropleth\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"type\": \"choropleth\"}], \"contour\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"contour\"}], \"contourcarpet\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"type\": \"contourcarpet\"}], \"heatmap\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"heatmap\"}], \"heatmapgl\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"heatmapgl\"}], \"histogram\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"histogram\"}], \"histogram2d\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"histogram2d\"}], \"histogram2dcontour\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"histogram2dcontour\"}], \"mesh3d\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"type\": \"mesh3d\"}], \"parcoords\": [{\"line\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"parcoords\"}], \"pie\": [{\"automargin\": true, \"type\": \"pie\"}], \"scatter\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatter\"}], \"scatter3d\": [{\"line\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatter3d\"}], \"scattercarpet\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scattercarpet\"}], \"scattergeo\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scattergeo\"}], \"scattergl\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scattergl\"}], \"scattermapbox\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scattermapbox\"}], \"scatterpolar\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatterpolar\"}], \"scatterpolargl\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatterpolargl\"}], \"scatterternary\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatterternary\"}], \"surface\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"surface\"}], \"table\": [{\"cells\": {\"fill\": {\"color\": \"#EBF0F8\"}, \"line\": {\"color\": \"white\"}}, \"header\": {\"fill\": {\"color\": \"#C8D4E3\"}, \"line\": {\"color\": \"white\"}}, \"type\": \"table\"}]}, \"layout\": {\"annotationdefaults\": {\"arrowcolor\": \"#2a3f5f\", \"arrowhead\": 0, \"arrowwidth\": 1}, \"coloraxis\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"colorscale\": {\"diverging\": [[0, \"#8e0152\"], [0.1, \"#c51b7d\"], [0.2, \"#de77ae\"], [0.3, \"#f1b6da\"], [0.4, \"#fde0ef\"], [0.5, \"#f7f7f7\"], [0.6, \"#e6f5d0\"], [0.7, \"#b8e186\"], [0.8, \"#7fbc41\"], [0.9, \"#4d9221\"], [1, \"#276419\"]], \"sequential\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"sequentialminus\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]]}, \"colorway\": [\"#636efa\", \"#EF553B\", \"#00cc96\", \"#ab63fa\", \"#FFA15A\", \"#19d3f3\", \"#FF6692\", \"#B6E880\", \"#FF97FF\", \"#FECB52\"], \"font\": {\"color\": \"#2a3f5f\"}, \"geo\": {\"bgcolor\": \"white\", \"lakecolor\": \"white\", \"landcolor\": \"#E5ECF6\", \"showlakes\": true, \"showland\": true, \"subunitcolor\": \"white\"}, \"hoverlabel\": {\"align\": \"left\"}, \"hovermode\": \"closest\", \"mapbox\": {\"style\": \"light\"}, \"paper_bgcolor\": \"white\", \"plot_bgcolor\": \"#E5ECF6\", \"polar\": {\"angularaxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}, \"bgcolor\": \"#E5ECF6\", \"radialaxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}}, \"scene\": {\"xaxis\": {\"backgroundcolor\": \"#E5ECF6\", \"gridcolor\": \"white\", \"gridwidth\": 2, \"linecolor\": \"white\", \"showbackground\": true, \"ticks\": \"\", \"zerolinecolor\": \"white\"}, \"yaxis\": {\"backgroundcolor\": \"#E5ECF6\", \"gridcolor\": \"white\", \"gridwidth\": 2, \"linecolor\": \"white\", \"showbackground\": true, \"ticks\": \"\", \"zerolinecolor\": \"white\"}, \"zaxis\": {\"backgroundcolor\": \"#E5ECF6\", \"gridcolor\": \"white\", \"gridwidth\": 2, \"linecolor\": \"white\", \"showbackground\": true, \"ticks\": \"\", \"zerolinecolor\": \"white\"}}, \"shapedefaults\": {\"line\": {\"color\": \"#2a3f5f\"}}, \"ternary\": {\"aaxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}, \"baxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}, \"bgcolor\": \"#E5ECF6\", \"caxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}}, \"title\": {\"x\": 0.05}, \"xaxis\": {\"automargin\": true, \"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\", \"title\": {\"standoff\": 15}, \"zerolinecolor\": \"white\", \"zerolinewidth\": 2}, \"yaxis\": {\"automargin\": true, \"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\", \"title\": {\"standoff\": 15}, \"zerolinecolor\": \"white\", \"zerolinewidth\": 2}}}, \"title\": {\"text\": \"2011 USA AGRICULTURAL EXPORTS\"}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('217aa6f5-7ba7-4f41-b1e4-9a7c773c7dfc');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "iplot(choromap)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "df3 = pd.read_csv('2014_World_GDP')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
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
       "      <th>COUNTRY</th>\n",
       "      <th>GDP (BILLIONS)</th>\n",
       "      <th>CODE</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>21.71</td>\n",
       "      <td>AFG</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Albania</td>\n",
       "      <td>13.40</td>\n",
       "      <td>ALB</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Algeria</td>\n",
       "      <td>227.80</td>\n",
       "      <td>DZA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>American Samoa</td>\n",
       "      <td>0.75</td>\n",
       "      <td>ASM</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Andorra</td>\n",
       "      <td>4.80</td>\n",
       "      <td>AND</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          COUNTRY  GDP (BILLIONS) CODE\n",
       "0     Afghanistan           21.71  AFG\n",
       "1         Albania           13.40  ALB\n",
       "2         Algeria          227.80  DZA\n",
       "3  American Samoa            0.75  ASM\n",
       "4         Andorra            4.80  AND"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = dict(type='choropleth',\n",
    "           locations=df3['CODE'],\n",
    "           z = df3['GDP (BILLIONS)'],\n",
    "           text = df3['COUNTRY'],\n",
    "           colorbar={'title':'GDP Billions US'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "layout = dict(title = '2014 Global GDP',\n",
    "             geo = dict(\n",
    "             showframe = False,\n",
    "             projection = {'type':'orthographic'}))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "choromap3 = go.Figure(data=[data], layout = layout)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "linkText": "Export to plot.ly",
        "plotlyServerURL": "https://plot.ly",
        "showLink": false
       },
       "data": [
        {
         "colorbar": {
          "title": {
           "text": "GDP Billions US"
          }
         },
         "locations": [
          "AFG",
          "ALB",
          "DZA",
          "ASM",
          "AND",
          "AGO",
          "AIA",
          "ATG",
          "ARG",
          "ARM",
          "ABW",
          "AUS",
          "AUT",
          "AZE",
          "BHM",
          "BHR",
          "BGD",
          "BRB",
          "BLR",
          "BEL",
          "BLZ",
          "BEN",
          "BMU",
          "BTN",
          "BOL",
          "BIH",
          "BWA",
          "BRA",
          "VGB",
          "BRN",
          "BGR",
          "BFA",
          "MMR",
          "BDI",
          "CPV",
          "KHM",
          "CMR",
          "CAN",
          "CYM",
          "CAF",
          "TCD",
          "CHL",
          "CHN",
          "COL",
          "COM",
          "COD",
          "COG",
          "COK",
          "CRI",
          "CIV",
          "HRV",
          "CUB",
          "CUW",
          "CYP",
          "CZE",
          "DNK",
          "DJI",
          "DMA",
          "DOM",
          "ECU",
          "EGY",
          "SLV",
          "GNQ",
          "ERI",
          "EST",
          "ETH",
          "FLK",
          "FRO",
          "FJI",
          "FIN",
          "FRA",
          "PYF",
          "GAB",
          "GMB",
          "GEO",
          "DEU",
          "GHA",
          "GIB",
          "GRC",
          "GRL",
          "GRD",
          "GUM",
          "GTM",
          "GGY",
          "GNB",
          "GIN",
          "GUY",
          "HTI",
          "HND",
          "HKG",
          "HUN",
          "ISL",
          "IND",
          "IDN",
          "IRN",
          "IRQ",
          "IRL",
          "IMN",
          "ISR",
          "ITA",
          "JAM",
          "JPN",
          "JEY",
          "JOR",
          "KAZ",
          "KEN",
          "KIR",
          "KOR",
          "PRK",
          "KSV",
          "KWT",
          "KGZ",
          "LAO",
          "LVA",
          "LBN",
          "LSO",
          "LBR",
          "LBY",
          "LIE",
          "LTU",
          "LUX",
          "MAC",
          "MKD",
          "MDG",
          "MWI",
          "MYS",
          "MDV",
          "MLI",
          "MLT",
          "MHL",
          "MRT",
          "MUS",
          "MEX",
          "FSM",
          "MDA",
          "MCO",
          "MNG",
          "MNE",
          "MAR",
          "MOZ",
          "NAM",
          "NPL",
          "NLD",
          "NCL",
          "NZL",
          "NIC",
          "NGA",
          "NER",
          "NIU",
          "MNP",
          "NOR",
          "OMN",
          "PAK",
          "PLW",
          "PAN",
          "PNG",
          "PRY",
          "PER",
          "PHL",
          "POL",
          "PRT",
          "PRI",
          "QAT",
          "ROU",
          "RUS",
          "RWA",
          "KNA",
          "LCA",
          "MAF",
          "SPM",
          "VCT",
          "WSM",
          "SMR",
          "STP",
          "SAU",
          "SEN",
          "SRB",
          "SYC",
          "SLE",
          "SGP",
          "SXM",
          "SVK",
          "SVN",
          "SLB",
          "SOM",
          "ZAF",
          "SSD",
          "ESP",
          "LKA",
          "SDN",
          "SUR",
          "SWZ",
          "SWE",
          "CHE",
          "SYR",
          "TWN",
          "TJK",
          "TZA",
          "THA",
          "TLS",
          "TGO",
          "TON",
          "TTO",
          "TUN",
          "TUR",
          "TKM",
          "TUV",
          "UGA",
          "UKR",
          "ARE",
          "GBR",
          "USA",
          "URY",
          "UZB",
          "VUT",
          "VEN",
          "VNM",
          "VGB",
          "WBG",
          "YEM",
          "ZMB",
          "ZWE"
         ],
         "text": [
          "Afghanistan",
          "Albania",
          "Algeria",
          "American Samoa",
          "Andorra",
          "Angola",
          "Anguilla",
          "Antigua and Barbuda",
          "Argentina",
          "Armenia",
          "Aruba",
          "Australia",
          "Austria",
          "Azerbaijan",
          "Bahamas, The",
          "Bahrain",
          "Bangladesh",
          "Barbados",
          "Belarus",
          "Belgium",
          "Belize",
          "Benin",
          "Bermuda",
          "Bhutan",
          "Bolivia",
          "Bosnia and Herzegovina",
          "Botswana",
          "Brazil",
          "British Virgin Islands",
          "Brunei",
          "Bulgaria",
          "Burkina Faso",
          "Burma",
          "Burundi",
          "Cabo Verde",
          "Cambodia",
          "Cameroon",
          "Canada",
          "Cayman Islands",
          "Central African Republic",
          "Chad",
          "Chile",
          "China",
          "Colombia",
          "Comoros",
          "Congo, Democratic Republic of the",
          "Congo, Republic of the",
          "Cook Islands",
          "Costa Rica",
          "Cote d'Ivoire",
          "Croatia",
          "Cuba",
          "Curacao",
          "Cyprus",
          "Czech Republic",
          "Denmark",
          "Djibouti",
          "Dominica",
          "Dominican Republic",
          "Ecuador",
          "Egypt",
          "El Salvador",
          "Equatorial Guinea",
          "Eritrea",
          "Estonia",
          "Ethiopia",
          "Falkland Islands (Islas Malvinas)",
          "Faroe Islands",
          "Fiji",
          "Finland",
          "France",
          "French Polynesia",
          "Gabon",
          "Gambia, The",
          "Georgia",
          "Germany",
          "Ghana",
          "Gibraltar",
          "Greece",
          "Greenland",
          "Grenada",
          "Guam",
          "Guatemala",
          "Guernsey",
          "Guinea-Bissau",
          "Guinea",
          "Guyana",
          "Haiti",
          "Honduras",
          "Hong Kong",
          "Hungary",
          "Iceland",
          "India",
          "Indonesia",
          "Iran",
          "Iraq",
          "Ireland",
          "Isle of Man",
          "Israel",
          "Italy",
          "Jamaica",
          "Japan",
          "Jersey",
          "Jordan",
          "Kazakhstan",
          "Kenya",
          "Kiribati",
          "Korea, North",
          "Korea, South",
          "Kosovo",
          "Kuwait",
          "Kyrgyzstan",
          "Laos",
          "Latvia",
          "Lebanon",
          "Lesotho",
          "Liberia",
          "Libya",
          "Liechtenstein",
          "Lithuania",
          "Luxembourg",
          "Macau",
          "Macedonia",
          "Madagascar",
          "Malawi",
          "Malaysia",
          "Maldives",
          "Mali",
          "Malta",
          "Marshall Islands",
          "Mauritania",
          "Mauritius",
          "Mexico",
          "Micronesia, Federated States of",
          "Moldova",
          "Monaco",
          "Mongolia",
          "Montenegro",
          "Morocco",
          "Mozambique",
          "Namibia",
          "Nepal",
          "Netherlands",
          "New Caledonia",
          "New Zealand",
          "Nicaragua",
          "Nigeria",
          "Niger",
          "Niue",
          "Northern Mariana Islands",
          "Norway",
          "Oman",
          "Pakistan",
          "Palau",
          "Panama",
          "Papua New Guinea",
          "Paraguay",
          "Peru",
          "Philippines",
          "Poland",
          "Portugal",
          "Puerto Rico",
          "Qatar",
          "Romania",
          "Russia",
          "Rwanda",
          "Saint Kitts and Nevis",
          "Saint Lucia",
          "Saint Martin",
          "Saint Pierre and Miquelon",
          "Saint Vincent and the Grenadines",
          "Samoa",
          "San Marino",
          "Sao Tome and Principe",
          "Saudi Arabia",
          "Senegal",
          "Serbia",
          "Seychelles",
          "Sierra Leone",
          "Singapore",
          "Sint Maarten",
          "Slovakia",
          "Slovenia",
          "Solomon Islands",
          "Somalia",
          "South Africa",
          "South Sudan",
          "Spain",
          "Sri Lanka",
          "Sudan",
          "Suriname",
          "Swaziland",
          "Sweden",
          "Switzerland",
          "Syria",
          "Taiwan",
          "Tajikistan",
          "Tanzania",
          "Thailand",
          "Timor-Leste",
          "Togo",
          "Tonga",
          "Trinidad and Tobago",
          "Tunisia",
          "Turkey",
          "Turkmenistan",
          "Tuvalu",
          "Uganda",
          "Ukraine",
          "United Arab Emirates",
          "United Kingdom",
          "United States",
          "Uruguay",
          "Uzbekistan",
          "Vanuatu",
          "Venezuela",
          "Vietnam",
          "Virgin Islands",
          "West Bank",
          "Yemen",
          "Zambia",
          "Zimbabwe"
         ],
         "type": "choropleth",
         "z": [
          21.71,
          13.4,
          227.8,
          0.75,
          4.8,
          131.4,
          0.18,
          1.24,
          536.2,
          10.88,
          2.52,
          1483,
          436.1,
          77.91,
          8.65,
          34.05,
          186.6,
          4.28,
          75.25,
          527.8,
          1.67,
          9.24,
          5.2,
          2.09,
          34.08,
          19.55,
          16.3,
          2244,
          1.1,
          17.43,
          55.08,
          13.38,
          65.29,
          3.04,
          1.98,
          16.9,
          32.16,
          1794,
          2.25,
          1.73,
          15.84,
          264.1,
          10360,
          400.1,
          0.72,
          32.67,
          14.11,
          0.18,
          50.46,
          33.96,
          57.18,
          77.15,
          5.6,
          21.34,
          205.6,
          347.2,
          1.58,
          0.51,
          64.05,
          100.5,
          284.9,
          25.14,
          15.4,
          3.87,
          26.36,
          49.86,
          0.16,
          2.32,
          4.17,
          276.3,
          2902,
          7.15,
          20.68,
          0.92,
          16.13,
          3820,
          35.48,
          1.85,
          246.4,
          2.16,
          0.84,
          4.6,
          58.3,
          2.74,
          1.04,
          6.77,
          3.14,
          8.92,
          19.37,
          292.7,
          129.7,
          16.2,
          2048,
          856.1,
          402.7,
          232.2,
          245.8,
          4.08,
          305,
          2129,
          13.92,
          4770,
          5.77,
          36.55,
          225.6,
          62.72,
          0.16,
          28,
          1410,
          5.99,
          179.3,
          7.65,
          11.71,
          32.82,
          47.5,
          2.46,
          2.07,
          49.34,
          5.11,
          48.72,
          63.93,
          51.68,
          10.92,
          11.19,
          4.41,
          336.9,
          2.41,
          12.04,
          10.57,
          0.18,
          4.29,
          12.72,
          1296,
          0.34,
          7.74,
          6.06,
          11.73,
          4.66,
          112.6,
          16.59,
          13.11,
          19.64,
          880.4,
          11.1,
          201,
          11.85,
          594.3,
          8.29,
          0.01,
          1.23,
          511.6,
          80.54,
          237.5,
          0.65,
          44.69,
          16.1,
          31.3,
          208.2,
          284.6,
          552.2,
          228.2,
          93.52,
          212,
          199,
          2057,
          8,
          0.81,
          1.35,
          0.56,
          0.22,
          0.75,
          0.83,
          1.86,
          0.36,
          777.9,
          15.88,
          42.65,
          1.47,
          5.41,
          307.9,
          304.1,
          99.75,
          49.93,
          1.16,
          2.37,
          341.2,
          11.89,
          1400,
          71.57,
          70.03,
          5.27,
          3.84,
          559.1,
          679,
          64.7,
          529.5,
          9.16,
          36.62,
          373.8,
          4.51,
          4.84,
          0.49,
          29.63,
          49.12,
          813.3,
          43.5,
          0.04,
          26.09,
          134.9,
          416.4,
          2848,
          17420,
          55.6,
          63.08,
          0.82,
          209.2,
          187.8,
          5.08,
          6.64,
          45.45,
          25.61,
          13.74
         ]
        }
       ],
       "layout": {
        "geo": {
         "projection": {
          "type": "orthographic"
         },
         "showframe": false
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "2014 Global GDP"
        }
       }
      },
      "text/html": [
       "<div>                            <div id=\"3200084e-1bfa-4b16-8e8c-cee27c291687\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"3200084e-1bfa-4b16-8e8c-cee27c291687\")) {                    Plotly.newPlot(                        \"3200084e-1bfa-4b16-8e8c-cee27c291687\",                        [{\"colorbar\": {\"title\": {\"text\": \"GDP Billions US\"}}, \"locations\": [\"AFG\", \"ALB\", \"DZA\", \"ASM\", \"AND\", \"AGO\", \"AIA\", \"ATG\", \"ARG\", \"ARM\", \"ABW\", \"AUS\", \"AUT\", \"AZE\", \"BHM\", \"BHR\", \"BGD\", \"BRB\", \"BLR\", \"BEL\", \"BLZ\", \"BEN\", \"BMU\", \"BTN\", \"BOL\", \"BIH\", \"BWA\", \"BRA\", \"VGB\", \"BRN\", \"BGR\", \"BFA\", \"MMR\", \"BDI\", \"CPV\", \"KHM\", \"CMR\", \"CAN\", \"CYM\", \"CAF\", \"TCD\", \"CHL\", \"CHN\", \"COL\", \"COM\", \"COD\", \"COG\", \"COK\", \"CRI\", \"CIV\", \"HRV\", \"CUB\", \"CUW\", \"CYP\", \"CZE\", \"DNK\", \"DJI\", \"DMA\", \"DOM\", \"ECU\", \"EGY\", \"SLV\", \"GNQ\", \"ERI\", \"EST\", \"ETH\", \"FLK\", \"FRO\", \"FJI\", \"FIN\", \"FRA\", \"PYF\", \"GAB\", \"GMB\", \"GEO\", \"DEU\", \"GHA\", \"GIB\", \"GRC\", \"GRL\", \"GRD\", \"GUM\", \"GTM\", \"GGY\", \"GNB\", \"GIN\", \"GUY\", \"HTI\", \"HND\", \"HKG\", \"HUN\", \"ISL\", \"IND\", \"IDN\", \"IRN\", \"IRQ\", \"IRL\", \"IMN\", \"ISR\", \"ITA\", \"JAM\", \"JPN\", \"JEY\", \"JOR\", \"KAZ\", \"KEN\", \"KIR\", \"KOR\", \"PRK\", \"KSV\", \"KWT\", \"KGZ\", \"LAO\", \"LVA\", \"LBN\", \"LSO\", \"LBR\", \"LBY\", \"LIE\", \"LTU\", \"LUX\", \"MAC\", \"MKD\", \"MDG\", \"MWI\", \"MYS\", \"MDV\", \"MLI\", \"MLT\", \"MHL\", \"MRT\", \"MUS\", \"MEX\", \"FSM\", \"MDA\", \"MCO\", \"MNG\", \"MNE\", \"MAR\", \"MOZ\", \"NAM\", \"NPL\", \"NLD\", \"NCL\", \"NZL\", \"NIC\", \"NGA\", \"NER\", \"NIU\", \"MNP\", \"NOR\", \"OMN\", \"PAK\", \"PLW\", \"PAN\", \"PNG\", \"PRY\", \"PER\", \"PHL\", \"POL\", \"PRT\", \"PRI\", \"QAT\", \"ROU\", \"RUS\", \"RWA\", \"KNA\", \"LCA\", \"MAF\", \"SPM\", \"VCT\", \"WSM\", \"SMR\", \"STP\", \"SAU\", \"SEN\", \"SRB\", \"SYC\", \"SLE\", \"SGP\", \"SXM\", \"SVK\", \"SVN\", \"SLB\", \"SOM\", \"ZAF\", \"SSD\", \"ESP\", \"LKA\", \"SDN\", \"SUR\", \"SWZ\", \"SWE\", \"CHE\", \"SYR\", \"TWN\", \"TJK\", \"TZA\", \"THA\", \"TLS\", \"TGO\", \"TON\", \"TTO\", \"TUN\", \"TUR\", \"TKM\", \"TUV\", \"UGA\", \"UKR\", \"ARE\", \"GBR\", \"USA\", \"URY\", \"UZB\", \"VUT\", \"VEN\", \"VNM\", \"VGB\", \"WBG\", \"YEM\", \"ZMB\", \"ZWE\"], \"text\": [\"Afghanistan\", \"Albania\", \"Algeria\", \"American Samoa\", \"Andorra\", \"Angola\", \"Anguilla\", \"Antigua and Barbuda\", \"Argentina\", \"Armenia\", \"Aruba\", \"Australia\", \"Austria\", \"Azerbaijan\", \"Bahamas, The\", \"Bahrain\", \"Bangladesh\", \"Barbados\", \"Belarus\", \"Belgium\", \"Belize\", \"Benin\", \"Bermuda\", \"Bhutan\", \"Bolivia\", \"Bosnia and Herzegovina\", \"Botswana\", \"Brazil\", \"British Virgin Islands\", \"Brunei\", \"Bulgaria\", \"Burkina Faso\", \"Burma\", \"Burundi\", \"Cabo Verde\", \"Cambodia\", \"Cameroon\", \"Canada\", \"Cayman Islands\", \"Central African Republic\", \"Chad\", \"Chile\", \"China\", \"Colombia\", \"Comoros\", \"Congo, Democratic Republic of the\", \"Congo, Republic of the\", \"Cook Islands\", \"Costa Rica\", \"Cote d'Ivoire\", \"Croatia\", \"Cuba\", \"Curacao\", \"Cyprus\", \"Czech Republic\", \"Denmark\", \"Djibouti\", \"Dominica\", \"Dominican Republic\", \"Ecuador\", \"Egypt\", \"El Salvador\", \"Equatorial Guinea\", \"Eritrea\", \"Estonia\", \"Ethiopia\", \"Falkland Islands (Islas Malvinas)\", \"Faroe Islands\", \"Fiji\", \"Finland\", \"France\", \"French Polynesia\", \"Gabon\", \"Gambia, The\", \"Georgia\", \"Germany\", \"Ghana\", \"Gibraltar\", \"Greece\", \"Greenland\", \"Grenada\", \"Guam\", \"Guatemala\", \"Guernsey\", \"Guinea-Bissau\", \"Guinea\", \"Guyana\", \"Haiti\", \"Honduras\", \"Hong Kong\", \"Hungary\", \"Iceland\", \"India\", \"Indonesia\", \"Iran\", \"Iraq\", \"Ireland\", \"Isle of Man\", \"Israel\", \"Italy\", \"Jamaica\", \"Japan\", \"Jersey\", \"Jordan\", \"Kazakhstan\", \"Kenya\", \"Kiribati\", \"Korea, North\", \"Korea, South\", \"Kosovo\", \"Kuwait\", \"Kyrgyzstan\", \"Laos\", \"Latvia\", \"Lebanon\", \"Lesotho\", \"Liberia\", \"Libya\", \"Liechtenstein\", \"Lithuania\", \"Luxembourg\", \"Macau\", \"Macedonia\", \"Madagascar\", \"Malawi\", \"Malaysia\", \"Maldives\", \"Mali\", \"Malta\", \"Marshall Islands\", \"Mauritania\", \"Mauritius\", \"Mexico\", \"Micronesia, Federated States of\", \"Moldova\", \"Monaco\", \"Mongolia\", \"Montenegro\", \"Morocco\", \"Mozambique\", \"Namibia\", \"Nepal\", \"Netherlands\", \"New Caledonia\", \"New Zealand\", \"Nicaragua\", \"Nigeria\", \"Niger\", \"Niue\", \"Northern Mariana Islands\", \"Norway\", \"Oman\", \"Pakistan\", \"Palau\", \"Panama\", \"Papua New Guinea\", \"Paraguay\", \"Peru\", \"Philippines\", \"Poland\", \"Portugal\", \"Puerto Rico\", \"Qatar\", \"Romania\", \"Russia\", \"Rwanda\", \"Saint Kitts and Nevis\", \"Saint Lucia\", \"Saint Martin\", \"Saint Pierre and Miquelon\", \"Saint Vincent and the Grenadines\", \"Samoa\", \"San Marino\", \"Sao Tome and Principe\", \"Saudi Arabia\", \"Senegal\", \"Serbia\", \"Seychelles\", \"Sierra Leone\", \"Singapore\", \"Sint Maarten\", \"Slovakia\", \"Slovenia\", \"Solomon Islands\", \"Somalia\", \"South Africa\", \"South Sudan\", \"Spain\", \"Sri Lanka\", \"Sudan\", \"Suriname\", \"Swaziland\", \"Sweden\", \"Switzerland\", \"Syria\", \"Taiwan\", \"Tajikistan\", \"Tanzania\", \"Thailand\", \"Timor-Leste\", \"Togo\", \"Tonga\", \"Trinidad and Tobago\", \"Tunisia\", \"Turkey\", \"Turkmenistan\", \"Tuvalu\", \"Uganda\", \"Ukraine\", \"United Arab Emirates\", \"United Kingdom\", \"United States\", \"Uruguay\", \"Uzbekistan\", \"Vanuatu\", \"Venezuela\", \"Vietnam\", \"Virgin Islands\", \"West Bank\", \"Yemen\", \"Zambia\", \"Zimbabwe\"], \"type\": \"choropleth\", \"z\": [21.71, 13.4, 227.8, 0.75, 4.8, 131.4, 0.18, 1.24, 536.2, 10.88, 2.52, 1483.0, 436.1, 77.91, 8.65, 34.05, 186.6, 4.28, 75.25, 527.8, 1.67, 9.24, 5.2, 2.09, 34.08, 19.55, 16.3, 2244.0, 1.1, 17.43, 55.08, 13.38, 65.29, 3.04, 1.98, 16.9, 32.16, 1794.0, 2.25, 1.73, 15.84, 264.1, 10360.0, 400.1, 0.72, 32.67, 14.11, 0.18, 50.46, 33.96, 57.18, 77.15, 5.6, 21.34, 205.6, 347.2, 1.58, 0.51, 64.05, 100.5, 284.9, 25.14, 15.4, 3.87, 26.36, 49.86, 0.16, 2.32, 4.17, 276.3, 2902.0, 7.15, 20.68, 0.92, 16.13, 3820.0, 35.48, 1.85, 246.4, 2.16, 0.84, 4.6, 58.3, 2.74, 1.04, 6.77, 3.14, 8.92, 19.37, 292.7, 129.7, 16.2, 2048.0, 856.1, 402.7, 232.2, 245.8, 4.08, 305.0, 2129.0, 13.92, 4770.0, 5.77, 36.55, 225.6, 62.72, 0.16, 28.0, 1410.0, 5.99, 179.3, 7.65, 11.71, 32.82, 47.5, 2.46, 2.07, 49.34, 5.11, 48.72, 63.93, 51.68, 10.92, 11.19, 4.41, 336.9, 2.41, 12.04, 10.57, 0.18, 4.29, 12.72, 1296.0, 0.34, 7.74, 6.06, 11.73, 4.66, 112.6, 16.59, 13.11, 19.64, 880.4, 11.1, 201.0, 11.85, 594.3, 8.29, 0.01, 1.23, 511.6, 80.54, 237.5, 0.65, 44.69, 16.1, 31.3, 208.2, 284.6, 552.2, 228.2, 93.52, 212.0, 199.0, 2057.0, 8.0, 0.81, 1.35, 0.56, 0.22, 0.75, 0.83, 1.86, 0.36, 777.9, 15.88, 42.65, 1.47, 5.41, 307.9, 304.1, 99.75, 49.93, 1.16, 2.37, 341.2, 11.89, 1400.0, 71.57, 70.03, 5.27, 3.84, 559.1, 679.0, 64.7, 529.5, 9.16, 36.62, 373.8, 4.51, 4.84, 0.49, 29.63, 49.12, 813.3, 43.5, 0.04, 26.09, 134.9, 416.4, 2848.0, 17420.0, 55.6, 63.08, 0.82, 209.2, 187.8, 5.08, 6.64, 45.45, 25.61, 13.74]}],                        {\"geo\": {\"projection\": {\"type\": \"orthographic\"}, \"showframe\": false}, \"template\": {\"data\": {\"bar\": [{\"error_x\": {\"color\": \"#2a3f5f\"}, \"error_y\": {\"color\": \"#2a3f5f\"}, \"marker\": {\"line\": {\"color\": \"#E5ECF6\", \"width\": 0.5}}, \"type\": \"bar\"}], \"barpolar\": [{\"marker\": {\"line\": {\"color\": \"#E5ECF6\", \"width\": 0.5}}, \"type\": \"barpolar\"}], \"carpet\": [{\"aaxis\": {\"endlinecolor\": \"#2a3f5f\", \"gridcolor\": \"white\", \"linecolor\": \"white\", \"minorgridcolor\": \"white\", \"startlinecolor\": \"#2a3f5f\"}, \"baxis\": {\"endlinecolor\": \"#2a3f5f\", \"gridcolor\": \"white\", \"linecolor\": \"white\", \"minorgridcolor\": \"white\", \"startlinecolor\": \"#2a3f5f\"}, \"type\": \"carpet\"}], \"choropleth\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"type\": \"choropleth\"}], \"contour\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"contour\"}], \"contourcarpet\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"type\": \"contourcarpet\"}], \"heatmap\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"heatmap\"}], \"heatmapgl\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"heatmapgl\"}], \"histogram\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"histogram\"}], \"histogram2d\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"histogram2d\"}], \"histogram2dcontour\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"histogram2dcontour\"}], \"mesh3d\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"type\": \"mesh3d\"}], \"parcoords\": [{\"line\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"parcoords\"}], \"pie\": [{\"automargin\": true, \"type\": \"pie\"}], \"scatter\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatter\"}], \"scatter3d\": [{\"line\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatter3d\"}], \"scattercarpet\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scattercarpet\"}], \"scattergeo\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scattergeo\"}], \"scattergl\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scattergl\"}], \"scattermapbox\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scattermapbox\"}], \"scatterpolar\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatterpolar\"}], \"scatterpolargl\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatterpolargl\"}], \"scatterternary\": [{\"marker\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"type\": \"scatterternary\"}], \"surface\": [{\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}, \"colorscale\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"type\": \"surface\"}], \"table\": [{\"cells\": {\"fill\": {\"color\": \"#EBF0F8\"}, \"line\": {\"color\": \"white\"}}, \"header\": {\"fill\": {\"color\": \"#C8D4E3\"}, \"line\": {\"color\": \"white\"}}, \"type\": \"table\"}]}, \"layout\": {\"annotationdefaults\": {\"arrowcolor\": \"#2a3f5f\", \"arrowhead\": 0, \"arrowwidth\": 1}, \"coloraxis\": {\"colorbar\": {\"outlinewidth\": 0, \"ticks\": \"\"}}, \"colorscale\": {\"diverging\": [[0, \"#8e0152\"], [0.1, \"#c51b7d\"], [0.2, \"#de77ae\"], [0.3, \"#f1b6da\"], [0.4, \"#fde0ef\"], [0.5, \"#f7f7f7\"], [0.6, \"#e6f5d0\"], [0.7, \"#b8e186\"], [0.8, \"#7fbc41\"], [0.9, \"#4d9221\"], [1, \"#276419\"]], \"sequential\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]], \"sequentialminus\": [[0.0, \"#0d0887\"], [0.1111111111111111, \"#46039f\"], [0.2222222222222222, \"#7201a8\"], [0.3333333333333333, \"#9c179e\"], [0.4444444444444444, \"#bd3786\"], [0.5555555555555556, \"#d8576b\"], [0.6666666666666666, \"#ed7953\"], [0.7777777777777778, \"#fb9f3a\"], [0.8888888888888888, \"#fdca26\"], [1.0, \"#f0f921\"]]}, \"colorway\": [\"#636efa\", \"#EF553B\", \"#00cc96\", \"#ab63fa\", \"#FFA15A\", \"#19d3f3\", \"#FF6692\", \"#B6E880\", \"#FF97FF\", \"#FECB52\"], \"font\": {\"color\": \"#2a3f5f\"}, \"geo\": {\"bgcolor\": \"white\", \"lakecolor\": \"white\", \"landcolor\": \"#E5ECF6\", \"showlakes\": true, \"showland\": true, \"subunitcolor\": \"white\"}, \"hoverlabel\": {\"align\": \"left\"}, \"hovermode\": \"closest\", \"mapbox\": {\"style\": \"light\"}, \"paper_bgcolor\": \"white\", \"plot_bgcolor\": \"#E5ECF6\", \"polar\": {\"angularaxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}, \"bgcolor\": \"#E5ECF6\", \"radialaxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}}, \"scene\": {\"xaxis\": {\"backgroundcolor\": \"#E5ECF6\", \"gridcolor\": \"white\", \"gridwidth\": 2, \"linecolor\": \"white\", \"showbackground\": true, \"ticks\": \"\", \"zerolinecolor\": \"white\"}, \"yaxis\": {\"backgroundcolor\": \"#E5ECF6\", \"gridcolor\": \"white\", \"gridwidth\": 2, \"linecolor\": \"white\", \"showbackground\": true, \"ticks\": \"\", \"zerolinecolor\": \"white\"}, \"zaxis\": {\"backgroundcolor\": \"#E5ECF6\", \"gridcolor\": \"white\", \"gridwidth\": 2, \"linecolor\": \"white\", \"showbackground\": true, \"ticks\": \"\", \"zerolinecolor\": \"white\"}}, \"shapedefaults\": {\"line\": {\"color\": \"#2a3f5f\"}}, \"ternary\": {\"aaxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}, \"baxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}, \"bgcolor\": \"#E5ECF6\", \"caxis\": {\"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\"}}, \"title\": {\"x\": 0.05}, \"xaxis\": {\"automargin\": true, \"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\", \"title\": {\"standoff\": 15}, \"zerolinecolor\": \"white\", \"zerolinewidth\": 2}, \"yaxis\": {\"automargin\": true, \"gridcolor\": \"white\", \"linecolor\": \"white\", \"ticks\": \"\", \"title\": {\"standoff\": 15}, \"zerolinecolor\": \"white\", \"zerolinewidth\": 2}}}, \"title\": {\"text\": \"2014 Global GDP\"}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('3200084e-1bfa-4b16-8e8c-cee27c291687');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "iplot(choromap3)"
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
