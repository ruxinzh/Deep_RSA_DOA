import argparse
import torch
import scipy.io
from eval_fun import run_monte_carlo_accuracy, run_monte_carlo_accuracy2, run_monte_carlo_sep, plot_results, plot_HR, model_complexity, generate_complex_signal, run_examples
def main():
    parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for DOA estimation accuracy")
    parser.add_argument('--num_simulations', type=int, default=100, help="Number of Monte Carlo simulations to run")
    parser.add_argument('--num_antennas', type=int, default=10, help="Number of antennas in the array")
    parser.add_argument('--evaluation_mode', type=str, default='accuracy1', help="Evaluation mode: accuracy1, accuracy2, separate, examples, complexity")
    parser.add_argument('--real', type=bool, default = True, help="using real world data demo")
    args = parser.parse_args()

    if args.evaluation_mode == 'accuracy1':
        # Accuracy - single target
        print('Monte Carlo simulation on single target case with ULA \n')
        snr_levels, mse_metrics = run_monte_carlo_accuracy(args.num_simulations, args.num_antennas, False)
        plot_results(snr_levels, mse_metrics)
        print('Monte Carlo simulation on single target case with SLA \n')
        snr_levels, mse_metrics = run_monte_carlo_accuracy(args.num_simulations, args.num_antennas)
        plot_results(snr_levels, mse_metrics)

    elif args.evaluation_mode == 'accuracy2':
        # Accuracy - two targets
        print('Monte Carlo simulation on two targets case with ULA \n')
        snr_levels, mse_metrics = run_monte_carlo_accuracy2(args.num_simulations, args.num_antennas, False)
        plot_results(snr_levels, mse_metrics)
        print('Monte Carlo simulation on two targets case with SLA \n')
        snr_levels, mse_metrics = run_monte_carlo_accuracy2(args.num_simulations, args.num_antennas)
        plot_results(snr_levels, mse_metrics)
        
    elif args.evaluation_mode == 'separate':
        # Sseparability
        print('Monte Carlo simulation on separability with ULA \n')
        sep_angles, HR_metrics = run_monte_carlo_sep(args.num_simulations, args.num_antennas, False)
        plot_HR(sep_angles, HR_metrics)
        print('Monte Carlo simulation on separability with SLA \n')
        sep_angles, HR_metrics = run_monte_carlo_sep(args.num_simulations, args.num_antennas)
        plot_HR(sep_angles, HR_metrics)
        
    elif args.evaluation_mode == 'examples':
        # Generate estimation example on real data
        if args.real:
            file_path = 'realData_demo.mat'
            mat_data = scipy.io.loadmat(file_path)
            bv = mat_data['bv_final']
            bv = bv[:,0:10].T
            signal = torch.from_numpy(bv).to(torch.cfloat).numpy()
        else:
            signal = generate_complex_signal(10, 40, torch.tensor([0, 7])).numpy()
        run_examples(signal, num_antennas=10, sparse_flag=False)
        run_examples(signal, num_antennas=10, sparse_flag=True)

    elif args.evaluation_mode == 'complexity':
        # Counts total trainable parameters
        model_complexity()

if __name__ == "__main__":
    main()