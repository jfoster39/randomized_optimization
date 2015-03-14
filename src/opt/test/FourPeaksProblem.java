package opt.test;

import java.util.Arrays;
import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

public class FourPeaksProblem {
    private static final int N = 80;

    private static final int T = N/10;

    // Helper function to calculate means
    public static double mean(double[] m) {
        double sum = 0;
        for (int i = 0; i < m.length; i++) {
            sum += m[i];
        }
        return sum / m.length;
    }

    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        long start;

        double[] rhc_runtime    = new double[10];
        double[] rhc_optimal    = new double[10];
        double[] sa_runtime     = new double[10];
        double[] sa_optimal     = new double[10];
        double[] ga_runtime     = new double[10];
        double[] ga_optimal     = new double[10];
        double[] mimic_runtime  = new double[10];
        double[] mimic_optimal  = new double[10];

        for (int i = 0; i < 10; i++) {
        	System.out.println("Iteration " + i);
	        start = System.currentTimeMillis();
	        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
	        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
	        fit.train();
	        System.out.println(ef.value(rhc.getOptimal()));
	        System.out.println("Runtime: " + (System.currentTimeMillis() - start));
            rhc_optimal[i] = ef.value(rhc.getOptimal());
            rhc_runtime[i] = System.currentTimeMillis() - start;

	        start = System.currentTimeMillis();
	        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
	        fit = new FixedIterationTrainer(sa, 200000);
	        fit.train();
	        System.out.println(ef.value(sa.getOptimal()));
	        System.out.println("Runtime: " + (System.currentTimeMillis() - start));
            sa_optimal[i] = ef.value(sa.getOptimal());
            sa_runtime[i] = System.currentTimeMillis() - start;

	        start = System.currentTimeMillis();
	        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
	        fit = new FixedIterationTrainer(ga, 1000);
	        fit.train();
	        System.out.println(ef.value(ga.getOptimal()));
	        System.out.println("Runtime: " + (System.currentTimeMillis() - start));
            ga_optimal[i] = ef.value(ga.getOptimal());
            ga_runtime[i] = System.currentTimeMillis() - start;

	        start = System.currentTimeMillis();
	        MIMIC mimic = new MIMIC(200, 5, pop);
	        fit = new FixedIterationTrainer(mimic, 1000);
	        fit.train();
	        System.out.println(ef.value(mimic.getOptimal()));
	        System.out.println("Runtime: " + (System.currentTimeMillis() - start));
	        System.out.println("\n\n");
            mimic_optimal[i] = ef.value(mimic.getOptimal());
            mimic_runtime[i] = System.currentTimeMillis() - start;
        }

        double rhc_runtime_mean     = mean( rhc_runtime );
        double rhc_optimal_mean     = mean( rhc_optimal );
        double sa_runtime_mean      = mean( sa_runtime  );
        double sa_optimal_mean      = mean( sa_optimal  );
        double ga_runtime_mean      = mean( ga_runtime  );
        double ga_optimal_mean      = mean( ga_optimal  );
        double mimic_runtime_mean   = mean( mimic_runtime );
        double mimic_optimal_mean   = mean( mimic_optimal );

        System.out.println( "Randomized Hill Climbing Results" );
        System.out.println( "Runtime: " + rhc_runtime_mean );
        System.out.println( "Optimal: " + rhc_optimal_mean + "\n" );

        System.out.println( "Simulated Annealing Results" );
        System.out.println( "Runtime: " + sa_runtime_mean );
        System.out.println( "Optimal: " + sa_optimal_mean + "\n" );

        System.out.println( "Genetic Algorithm Results" );
        System.out.println( "Runtime: " + ga_runtime_mean );
        System.out.println( "Optimal: " + ga_optimal_mean + "\n" );

        System.out.println( "MIMIC Results" );
        System.out.println( "Runtime: " + mimic_runtime_mean );
        System.out.println( "Optimal: " + mimic_optimal_mean + "\n" );
    }
}
