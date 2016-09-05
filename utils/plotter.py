import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

class Plotter():

    def __init__():
        return

    def plot_predicted_series(self, lstm_model, data_mat, resp_mat, time_vec,
                              it_num, base_path):
        plt.plot(time_vec, lstm_model.predict(data_mat)[0], marker='.',
                 label='Predicted')
        plt.plot(time_vec, resp_mat, marker='.', label='Actual')
        plt.title('Predicted plots after' + str(it_num) +
                  'training iterations')
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.savefig(base_path+'Iteration_' + str(it_num) + '.png')
        plt.close()

    def plot_error(self, error_df, n_prev, noiseSD, base_path, Training=True):
        if Training is True:
            str_val = 'Training'
        else:
            str_val = 'Test'
        plt.plot(error_df.index, error_df.Error)
        plt.title(str_val+' Error variation with Iterations')
        plt.xlabel('Iterations')
        plt.ylabel(str_val+' Error')
        plt.savefig(base_path+'error_n_prev_'+str(n_prev)+'_noise_' +
                    str(noiseSD)+'.png')
        plt.close()
