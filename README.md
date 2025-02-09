# FakeNewsDetection_MSc_thesis


<table>
<tr>
    <td rowspan="2"> </td>
    <td rowspan="2">name</td>
    <td colspan="5">2-way</td>
    <td colspan="5">3-way</td>
    <td colspan="5">6-way</td>
</tr>
<tr>
    <td> acc </td>
    <td> f1 </td>
    <td> recall </td>
    <td> precision </td>
    <td> FPR </td>
    <td> acc </td>
    <td> f1 </td>
    <td> recall </td>
    <td> precision </td>
    <td> FPR </td>
    <td> acc </td>
    <td> f1 </td>
    <td> recall </td>
    <td> precision </td>
    <td> FPR </td>
</tr>
<tr>
    <td rowspan="2">text</td>
    <td>bert</td>
    <td>88.54%</td>
    <td>88.02%</td>
    <td>88.14%</td>
    <td>87.92%</td>
    <td>10.00%</td>
    <td>88.03%</td>
    <td>87.46%</td>
    <td>85.79%</td>
    <td>89.39%</td>
    <td>9.95%</td>
    <td>81.35%</td>
    <td>74.29%</td>
    <td>70.28%</td>
    <td>80.85%</td>
    <td>11.56%</td>

</tr>
<tr>
    <td>clip-text</td>
    <td>80.61%</td>
    <td>79.97%</td>
    <td>80.43%</td>
    <td>79.70%</td>
    <td>18.73%</td>
    <td>74.08%</td>
    <td>62.15%</td>
    <td>74.68%</td>
    <td>59.80%</td>
    <td>22.78%</td>
    <td>55.83%</td>
    <td>37.93%</td>
    <td>43.74%</td>
    <td>36.62%</td>
    <td>24.07%</td>

</tr>
<tr>
    <td rowspan="2">viison</td>
    <td>vit</td>
    <td>83.98%</td>
    <td>83.52%</td>
    <td>84.28%</td>
    <td>83.18</td>
    <td>17.13%</td>
    <td>83.96%</td>
    <td>83.83%</td>
    <td>83.45%</td>
    <td>84.31%</td>
    <td>14.71%</td>
    <td>81.00%</td>
    <td>67.34%</td>
    <td>64.73%</td>
    <td>75.17%</td>
    <td>17.04%</td>
</tr>
<tr>
    <td>clip-vision</td>
    <td>60.60%</td>
    <td>37.73%</td>
    <td>50%</td>
    <td>30.29%</td>
    <td>x%</td>
    <td>43.09%</td>
    <td>26.78%</td>
    <td>44.32%</td>
    <td>35.21%</td>
    <td>3.60%</td>
    <td>39.40%</td>
    <td>56.53%</td>
    <td>100%</td>
    <td>39.40%</td>
    <td>100%</td>
</tr>

<tr>
    <td rowspan="3"> multi-modal</td>
    <td>clip</td>
    <td>80.66%</td>
    <td>80.04%</td>
    <td>80.55%</td>
    <td>79.76%</td>
    <td>18.92%</td>
    <td>61.78%</td>
    <td>50.72%</td>
    <td>66.55%</td>
    <td>50.89%</td>
    <td>29.83%</td>
    <td>55.87%</td>
    <td>44.11%</td>
    <td>56.04%</td>
    <td>42.32%</td>
    <td>21.59%</td>
</tr>
<tr>
    <td>bert-vit</td>
    <td>89.45%</td>
    <td>89.18%</td>
    <td>90.23%</td>
    <td>88.77%</td>
    <td>13.44%</td>
    <td>80.46%</td>
    <td>75.57%</td>
    <td>83.98%</td>
    <td>71.14%</td>
    <td>18.51%</td>
    <td>86.23%</td>
    <td>79.79%</td>
    <td>82.87%</td>
    <td>77.66%</td>
    <td>4.25%</td>
</tr>
<tr>
    <td>albef</td>
    <td>84.83%</td>
    <td>84.46%</td>
    <td>85.45%</td>
    <td>84.13%</td>
    <td>17.46%</td>
    <td>83.43%</td>
    <td>79.75%</td>
    <td>87.44%</td>
    <td>75.40%</td>
    <td>17.28%</td>
    <td>73.57%</td>
    <td>63.05%</td>
    <td>71.61%</td>
    <td>59.81%</td>
    <td>8.12%</td>
</tr>
</table>

|           | 2-way           | 0       | 1      |
|-----------|-----------------|---------|--------|
| text-only | chatGPT         | 24/50   | 37/50  |
| txt-vis   | 通义千问            | 9/15    | 11/15  |
| txt-vis   | miniGPT4        | 4/14    | 13/14  |
| txt-vis   | Eagle           | 9/11    | 5/14   |
| txt-vis   | MM-REACT        | 10/14   | 11/15  |
| txt-vis   | MiniCPM         | 12/15   | 6/12   |
| txt-vis   | LLaVA-onevision | 9/15    | 13/15  |


|           | 3-way           | 0      | 1     | 2      |
|-----------|-----------------|--------|-------|--------|
| text-only | chatGPT         | 43/50  | 0/50  | 20/50  |
| txt-vis   | 通义千问            | 8/15   | 7/15  | 6/15   |
| txt-vis   | miniGPT4        | 2/6    | 2/8   | 9/10   |
| txt-vis   | Eagle           | 4/15   | 5/10  | 13/14  |
| txt-vis   | MM-REACT        | 8/15   | 2/14  | 8/14   |
| txt-vis   | MiniCPM         | 0/15   | 2/5   | 6/8    |
| txt-vis   | LLaVA-onevision | 11/15  | 8/15  | 8/15   |

|            | 6-way            | 0      | 1     | 2      | 3     | 4     | 5     |
|------------|------------------|--------|-------|--------|-------|-------|-------|
| text-only  | chatGPT          | 42/50  | 23/50 | 4/50   | 0/50  | 3/50  | 1/50  |
| txt-vis    | 通义千问             | 12/15  | 1/15  | 0/15   | 0/15  | 3/15  | 0/15  |
| txt-vis    | miniGPT4         | 0/15   | 1/15  | 14/15  | 0/15  | 1/15  | 0/15  |
| txt-vis    | Eagle            | 15/15  | 2/14  | 1/14   | 0/15  | 0/15  | 0/15  |
| txt-vis    | MM-REACT         | 9/15   | 0/15  | 0/15   | 0/15  | 9/15  | 0/15  |
| txt-vis    | MiniCPM          | 1/6    | 1/6   | 0/1    | 0/7   | 2/5   | 0/5   |
| txt-vis    | LLaVA-onevision  | 11/15  | 1/5   | 4/15   | 0/15  | 0/15  | 1/15  |
