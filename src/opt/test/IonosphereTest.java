package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network
 */
public class IonosphereTest {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 34, hiddenLayer = 5, outputLayer = 1, trainingIterations = 500;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"Randomized Hill Climbing", "Simulated Annealing", "Genetic Algorithm"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        double[][] correct_classifications = new double[3][5];
        double[][] training_times          = new double[3][5];
        double[][] testing_times           = new double[3][5];

        for( int x = 0; x < 5; x++ )
        {
            int iteration_number = x + 1;
            results += "\n ITERATION " + iteration_number + " ************************************\n";

            for(int i = 0; i < oa.length; i++) {
                networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
                nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
            }

            oa[0] = new RandomizedHillClimbing(nnop[0]);
            oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
            oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

            for(int i = 0; i < oa.length; i++) {
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                train(oa[i], networks[i], oaNames[i]); //trainer.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10,9);

                Instance optimalInstance = oa[i].getOptimal();
                networks[i].setWeights(optimalInstance.getData());

                double predicted, actual;
                start = System.nanoTime();
                for(int j = 0; j < instances.length; j++) {
                    networks[i].setInputValues(instances[j].getData());
                    networks[i].run();

                    predicted = Double.parseDouble(instances[j].getLabel().toString());
                    actual = Double.parseDouble(networks[i].getOutputValues().toString());

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10,9);


                results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                            "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                            + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                            + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

                correct_classifications[i][x]   = (correct/(correct+incorrect)*100);
                training_times[i][x]            = trainingTime;
                testing_times[i][x]             = testingTime;
            }
        }

        // Calculate mean values for output
        double[] means = new double[3];
        for(int i = 0; i < 3; i++){
            double sum_correct_classification   = 0.0;
            double sum_training_times           = 0.0;
            double sum_testing_times            = 0.0;

            for (int j = 0; j < 5; j++){
                sum_correct_classification  += correct_classifications[i][j];
                sum_training_times          += training_times[i][j];
                sum_testing_times           += testing_times[i][j];
            }
            double mean_correct_classification  = sum_correct_classification / 5;
            double mean_training_time           = sum_training_times / 5;
            double mean_testing_time            = sum_testing_times / 5;

            results +=  "\n MEAN RESULTS OVER 5 ITERATIONS ************ " +
                        "\nResults for " + oaNames[i] + " instances.\nPercent correctly classified: "
                        + df.format(mean_correct_classification) + "%\nTraining time: " + df.format(mean_training_time)
                        + " seconds\nTesting time: " + df.format(mean_testing_time) + " seconds\n";
        }

        System.out.println( results );
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[351][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/ionosphere.txt")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[34]; // 10 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 34; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                String instanceClass = scan.next();
                double binaryClass = instanceClass.equals("g") ? 0 : 1;

                attributes[i][1][0] = binaryClass;
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications are either 0 or 1 -> good or bad i.e. g or b
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}
