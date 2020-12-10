
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import pyflux as pf
import warnings

warnings.filterwarnings("ignore")

def ecdf(ser, thresh):

    """
    compute the empirical cumulative distribution function of the series ser, with cutoff array thresh

    """

    l = len(ser)

    return pd.Series(data=[(float((ser <= x).sum())) / l for x in thresh], index=thresh)


def l_infty(cdf1, cdf2):

    hulp = pd.DataFrame()
    hulp['cdf1'] = (cdf1.reset_index()[0])
    hulp['cdf2'] = (cdf2.reset_index()[0])
    hulp['absdiff'] = (hulp['cdf1'] - hulp['cdf2']).abs()

    return hulp['absdiff'].max()


def c(alpha):

    return np.sqrt(-0.5 * np.log(alpha / 2))


def ks(n, m, alpha):

    return c(alpha) * np.sqrt((float(n + m)) / (n * m))


def get_CI(series, ci):
    
    """
    calculate lower and upper bound at confidence level ci
            
    """
    
    lower_bound=np.percentile(series,  (100-ci)/2)
    upper_bound=np.percentile(series,  ci+(100-ci)/2)
    
    return lower_bound, upper_bound


def merge_partitions(partitions, list_num_par):

    '''

    merge dataframe partitions

    Args:

        partitions (dict): dict of dataframes
        list_num_par (list of int): list of dataframe indexes to merge

    Returns:

        df (pandas.DataFrame): merged df

    '''

    df = partitions[0]

    for num_par in list_num_par[1:]:

        df=df.append(partitions[num_par])

    return df

    
class KSTest:

    """
    perform ks test on strategy performance series
    
    CI (float): confidence level

    """

    def __init__(self,
                 CI):
        
        self.maxDif = None
        self.KS_stat = None
        self.pValue = None
        self.cdf1 = None
        self.cdf2 = None
        self.CI = CI
        self.train_period = None
        self.valid_period = None
        self.performance_measure = None
        self.maxDif = None
        self.critical_Val = None
       
        return

    def perform_test(self,
                     data_Dir,
                     fileName,
                     Training_Period_Start,
                     Training_Period_End,
                     Validation_Period_Start,
                     Validation_Period_End,
                     performanceMeasure,
                     ):
        
        self.train_period = '{} to {}'.format(Training_Period_Start, Training_Period_End)
        self.valid_period = '{} to {}'.format(Validation_Period_Start, Validation_Period_End)
        self.performance_measure = performanceMeasure
    
        c2cData = pd.read_csv(data_Dir + fileName)
        dailyReturnDF = c2cData[['Date', performanceMeasure]]

        train_ret = dailyReturnDF[
            (dailyReturnDF['Date'] >= Training_Period_Start) & (dailyReturnDF['Date'] <= Training_Period_End)]

        valid_ret = dailyReturnDF[
            (dailyReturnDF['Date'] >= Validation_Period_Start) & (dailyReturnDF['Date'] <= Validation_Period_End)]

        rvs1 = train_ret[performanceMeasure].values
        rvs2 = valid_ret[performanceMeasure].values
        
        thresh = np.linspace(-2.5, 2.5, 10000)
        self.cdf1 = ecdf(rvs1, thresh)
        self.cdf2 = ecdf(rvs2, thresh)

        self.maxDif = l_infty(self.cdf1, self.cdf2)
        self.critical_Val = ks(len(rvs1), len(rvs2), 1 - self.CI)
        
        print('########################################')
        print('Approximate KS-test Values: left_hand_{}_RightHand_{}'.format(self.maxDif, self.critical_Val))

        if self.maxDif > self.critical_Val:
            print('Null Hypothesis is rejected and thus validation is not the same as training')
        else:
            print('Null Hypothesis is accepted and thus validation is the same as training')

        print('########################################')

        print('                                        ')
        print('########################################')

        print('SCIPY ks two-sided test')
        self.KS_stat, self.pValue = stats.ks_2samp(rvs1, rvs2)
        print('KS-test: KS_stat_{}_pValue_{}'.format(self.KS_stat, self.pValue))
        
        return

    def plot_result(self):
        
        from textwrap import wrap
        
        plt.figure()
        plt.plot(self.cdf1)
        plt.plot(self.cdf2)
        plt.gca().set_title("\n".join(wrap('KS Test, P-Value={} \
                            train_period: {} \
                            valid_period: {} \
                            performance_measure: {}' \
                            .format(round(self.pValue, 2), 
                                    self.train_period, self.valid_period, self.performance_measure), 60)),
                            fontsize='large')
                            
        plt.gca().legend(('train', 'valid'), fontsize='large')

        return


class BootstrapValid:
    
    """
    consistency check train data set and validation data set
    
    """
    
    def __init__(self):
        
        self.train_period = None
        self.valid_period = None
        self.performance_measure = None
        self.partition_size = None
        self.num_sample = None
        self.perf_metric = None
        self.dict_perf_metric_func={'annualised_sharpe': lambda df: df.mean(axis=0)/df.std(axis=0)*(252**0.5)}
        self.df_simulaiton = None
        self.train_series = None
        self.valid_series = None
        self.valid_result = None
        self.replacement = None
        
        return
    
             
    def simulate(self,  
                 data_Dir,
                 fileName,
                 Training_Period_Start,
                 Training_Period_End,
                 Validation_Period_Start,
                 Validation_Period_End,
                 performanceMeasure,
                 partition_size=3,
                 num_sample=1000,
                 perf_metric='annualised_sharpe',
                 replacement=True):
        
        self.train_period = '{} to {}'.format(Training_Period_Start, Training_Period_End)
        self.valid_period = '{} to {}'.format(Validation_Period_Start, Validation_Period_End)
        self.performance_measure = performanceMeasure
        self.partition_size = partition_size
        self.num_sample = num_sample
        self.perf_metric = perf_metric
        perf_metric_func=self.dict_perf_metric_func[self.perf_metric]
        df_return_series=pd.read_csv(data_Dir+fileName, index_col=0)[performanceMeasure]
        self.train_series=df_return_series.ix[Training_Period_Start:Training_Period_End]
        self.valid_series=df_return_series.ix[Validation_Period_Start:Validation_Period_End]
        self.replacement = replacement
        self.valid_result=perf_metric_func(self.valid_series)

        train_num_partition=int(self.train_series.shape[0]/partition_size)
        valid_num_partition=int(self.valid_series.shape[0]/partition_size)
        
        partitions=np.array_split(self.train_series,train_num_partition)
        
        list_partitions=list(range(train_num_partition))
        
        self.df_simulaiton=pd.DataFrame(index=range(num_sample), columns=['measurement'])
        
        for i in range(num_sample):
            
            simulated_valid_partition_num=np.random.choice(list_partitions, valid_num_partition, replace=self.replacement)
            
            df_simulated_valid=merge_partitions(partitions, simulated_valid_partition_num)
            
            self.df_simulaiton.ix[i]=perf_metric_func(df_simulated_valid)
        
        return
    
    
    def plot_result(self):
        
        """
        plot simulation result
        """
        
        lower_bound_95, upper_bound_95 = get_CI(self.df_simulaiton['measurement'], 95)
        lower_bound_90, upper_bound_90 = get_CI(self.df_simulaiton['measurement'], 90)
        
        self.df_simulaiton['measurement']=self.df_simulaiton['measurement'].astype(float)
        sns.kdeplot(self.df_simulaiton['measurement'], shade=True);
        plt.axvline(lower_bound_95, 0, 1, color='Red',linestyle='--',label='95% CI', linewidth=3)
        plt.axvline(upper_bound_95, 0, 1, color='Red',linestyle='--',label='95% CI', linewidth=3)
        
        plt.axvline(lower_bound_90, 0, 1, color='Green',linestyle='--',label='90% CI', linewidth=3)
        plt.axvline(upper_bound_90, 0, 1, color='Green',linestyle='--',label='90% CI', linewidth=3)
        
        plt.axvline(self.valid_result, 0, 1, color='Black',linestyle='--',label='Valid', linewidth=5)
        
        plt.title('num_sample = %s, %s, replacement=%s'%(self.num_sample, self.perf_metric, str(self.replacement)), size='large')
        plt.margins(0.02)
        plt.xlabel('perf_metric')
        plt.ylabel('pdf')
        
        plt.legend()
        plt.show()

        return

class validation_test_overview:
    
    """
    overview of the series
    
    """
    
    def __init__(self,
                 data_Dir,
                 fileName,
                 Training_Period_Start,
                 Training_Period_End,
                 Validation_Period_Start,
                 Validation_Period_End,
                 performanceMeasure):
        
        self.data_Dir = data_Dir,
        self.fileName = fileName,
        self.Training_Period_Start = Training_Period_Start,
        self.Training_Period_End = Training_Period_End,
        self.Validation_Period_Start = Validation_Period_Start,
        self.Validation_Period_End = Validation_Period_End,
        self.performance_measure  = performanceMeasure       
        self.train_period = '{} to {}'.format(Training_Period_Start, Training_Period_End)
        self.valid_period = '{} to {}'.format(Validation_Period_Start, Validation_Period_End)       
        self.df_return_series = pd.read_csv(data_Dir+fileName, index_col=0)[performanceMeasure]
        self.train_series = self.df_return_series.ix[Training_Period_Start:Training_Period_End]
            
    def plot_series(self):
        
        """
        plot overview of the series
        """
        
        self.df_return_series.index=pd.to_datetime(self.df_return_series.index)
        self.df_return_series.cumsum().plot()
        plt.axvline(self.Training_Period_End, 0, 1, color='Red',linestyle='--',label='train_valid_split', linewidth=3)
        plt.title('cumulative %s'%self.performance_measure)
        plt.legend()
        plt.show()        

        return
    

    def plot_acf(self):
        
        """
        check serial correlation
        """
        
        pf.acf_plot(self.train_series.dropna().values)
        
        return    
    
def generate_validation_report(dir_out, valid_test, ks_test, bootstrap_valid):
    
    """
    save report 
    
    dir_out (path)
    valid_test (instance)
    ks_test (instance)
    bootstrap_valid (instance)
    
    """
    
    # save all plots
    
    dir_plots=os.path.dirname(dir_out)+'/plot/'
    if not os.path.exists(dir_plots): os.makedirs(dir_plots)

    plt.close("all")
    valid_test.plot_series()
    plt.savefig(dir_plots+'series.png', bbox_inches='tight')
    plt.close("all")
    
    plt.close("all")
    valid_test.plot_acf()
    plt.savefig(dir_plots+'acf.png', bbox_inches='tight')
    plt.close("all")
        
    plt.close("all")
    bootstrap_valid.plot_result()
    plt.savefig(dir_plots+'bootstrap_valid.png', bbox_inches='tight')
    plt.close("all")
    
    # summary
    
    writer = pd.ExcelWriter(dir_out, engine='xlsxwriter')

    # overview 
    df_overview_param=pd.DataFrame(data=[str(valid_test.data_Dir), str(valid_test.fileName), valid_test.train_period,valid_test.valid_period, valid_test.performance_measure],
                                  index=['data_dir', 'file_name', 'train_period','valid_period','performance_measure'],
                                  columns=['value'])
    
    df_overview_param.index.names=['parameter']
    df_overview_param.to_excel(writer, sheet_name='parameters')
    
    worksheet = writer.sheets['parameters']
    worksheet.insert_image('H2', dir_plots+'series.png')
    worksheet.insert_image('H23', dir_plots+'acf.png')    
    
    # ks test
    
    description = "Computes the Kolmogorov-Smirnov statistic on 2 samples. \
    This is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous distribution."
    
    if ks_test.pValue < 1 - ks_test.CI:
    
        conclusion = 'Reject null hypothesis given p-value < 1 - CI'  
    
    else:
        
        conclusion = 'Can not reject null hypothesis given p-value >= 1 - CI'                
    
    df_ks_param=pd.DataFrame([ks_test.CI, ks_test.critical_Val, ks_test.maxDif, ks_test.pValue, description, conclusion],
                              index=['CI', 'critical_val','maxDif','pValue','description','conclusion'], 
                              columns=['value'])
        
    df_ks_param.index.names=['parameter']
    
    df_ks_param.to_excel(writer, sheet_name='ks_test_parameters')
        
    df_cdf1=pd.DataFrame(ks_test.cdf1)
    df_cdf1.columns=['cdf_train']
    df_cdf1.to_excel(writer, sheet_name='ks_test_cdf_train')
    
    df_cdf2=pd.DataFrame(ks_test.cdf2)
    df_cdf2.columns=['cdf_valid']
    df_cdf2.to_excel(writer, sheet_name='ks_test_cdf_valid')
        
    # insert chart
    chart = writer.book.add_chart({'type': 'line'})
    _shape=df_cdf1.shape
    
    chart.add_series({
        'name':       '=ks_test_cdf_train!$B$1',
        'categories': '=ks_test_cdf_train!$A$1:$A$%s'%_shape[0],
        'values':     '=ks_test_cdf_train!$B$1:$B$%s'%(_shape[0]),
    })

    chart.add_series({
        'name':       '=ks_test_cdf_valid!$B$1',
        'categories': '=ks_test_cdf_valid!$A$1:$A$%s'%_shape[0],
        'values':     '=ks_test_cdf_valid!$B$1:$B$%s'%(_shape[0]),
    })
        
    chart.set_title ({'name': 'KS Test'})
    chart.set_x_axis({'name': ''})
    chart.set_y_axis({'name': 'cdf'})
    chart.set_style(10)
    
    writer.sheets['ks_test_parameters'].insert_chart('H2', chart)

    # boostrap simulation test
    
    df_simula_param=pd.DataFrame([bootstrap_valid.perf_metric, bootstrap_valid.partition_size,
                                  bootstrap_valid.num_sample, bootstrap_valid.replacement ],
                                  index=['perf_metric','partition_size','num_sample','replacement'],
                                  columns=['value'])    
        
    df_simula_param.index.names=['parameter']
    df_simula_param.to_excel(writer, sheet_name='bootstrap_test')
    
    worksheet = writer.sheets['bootstrap_test']
    worksheet.insert_image('H2', dir_plots+'bootstrap_valid.png')
    
    writer.save()
    
    return
         
if __name__ == '__main__':

    pass

#   test    
    
#    data_Dir = '//farmnas/FARM2/Research/suhinthanm/TickData/Validation/'
#    fileName = '20181120_165127_C2C_report_NET_Tab.csv'
#    Training_Period_Start = '2015-01-01'
#    Training_Period_End = '2017-06-30'
#    Validation_Period_Start = '2017-07-01'
#    Validation_Period_End = '2018-06-30'
#    performanceMeasure = 'Return WITHOUT_COST'
#    
#    np.random.seed(32)
#
#    valid_test=validation_test_overview(data_Dir,
#                 fileName,
#                 Training_Period_Start,
#                 Training_Period_End,
#                 Validation_Period_Start,
#                 Validation_Period_End,
#                 performanceMeasure)
#        
#    ks_test = KSTest(CI=0.95)
#
#    ks_test.perform_test(data_Dir,
#                         fileName,
#                         Training_Period_Start,
#                         Training_Period_End,
#                         Validation_Period_Start,
#                         Validation_Period_End,
#                         performanceMeasure,
#                         )
#
#    bootstrap_valid = BootstrapValid ()
#
#    bootstrap_valid.simulate(data_Dir,
#                             fileName,
#                             Training_Period_Start,
#                             Training_Period_End,
#                             Validation_Period_Start,
#                             Validation_Period_End,
#                             performanceMeasure,
#                             partition_size=3,
#                             num_sample=100,
#                             perf_metric='annualised_sharpe')
#    
#    bootstrap_valid.plot_result()
# 
#    dir_out='//farmnas/FARM2/Research/mchen/reports/validation/validation_report_test.xlsx'
#   
#    generate_validation_report(dir_out, valid_test, ks_test, bootstrap_valid)
     