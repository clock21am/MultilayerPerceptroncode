import java.io.*;
import java.util.*;
public class mlp {
	
	public static void main(String[] args) {
		System.out.print("The goal of backpropagation is to optimize the weights \nso that the neural network can learn how to \ncorrectly map arbitrary inputs to outputs.");
		Scanner s = new Scanner(System.in);
		System.out.print("Enter the value of no of input nodes ");
		int nofi = s.nextInt();
		System.out.print("Enter the value of no of hidden nodes ");
		int nofo = s.nextInt();
		System.out.print("Enter the value of no of output nodes ");
		int nofh = s.nextInt();
		double[][] wih = new double[nofi][nofh];
		double[][] who = new double[nofh][nofo];
		double b1;
		b1 = s.nextDouble();
		double b2;
		b2 = s.nextDouble();
		double eta = 0.5;
		for(int i=0;i<nofi;i++){
			for(int j=0;j<nofh;i++){
				 wih[i][j] = s.nextDouble();
			}
		}
		for(int i=0;i<nofh;i++){
			for(int j=0;j<nofo;j++){
				who[i][j] = s.nextDouble();
			}
		}
		double traini[] = new double[nofi];
		double traino[] = new double[nofo];
		for(int i=0;i<nofi;i++){
			traini[i] = s.nextDouble();
		}
		for(int i=0;i<nofo;i++){
			traino[i] = s.nextDouble();
		}
		//forward pass 
		double[] neth = new double[nofh];
		for(int j=0;j<nofh;j++) {
		   for(int i=0;i<nofi;i++) {
			   neth[j] += wih[i][j]*traini[i];
		   }
		   neth[j] += b1*(j+1);
		   //when u apply the logistic function to get the output of h1
		   neth[j] = (1/( 1 + Math.pow(Math.E,(-1*neth[j]))));
		}
		// here we do the same for in order to get the ouptut 
		double[] neto = new double[nofo];
		for(int j=0;j<nofo;j++) {
			for(int i=0;i<nofh;i++) {
				neto[j] += who[j][i]*neth[j];
			}
			neto[j] =  (1/( 1 + Math.pow(Math.E,(-1*neth[j]))));
		}
		double[] error = new double[nofo];
		double two = 2.00;
		double totalerror = 0.00;
		
		//calculating total error
		for(int i=0;i<nofo;i++) {
			error[i] = ((1/2) * Math.pow(traino[i]-neto[i], two));
			totalerror += error[i];
		}
		//Backward pass
		//Our goal with backpropagation is to update each of the weights in the network so that they cause the actual output to be closer the target output,
		//thereby minimizing the error for each output neuron and the network as a whole.
		
		
		//Applying delta rule in the output layer
		double[] oe1 = new double[nofo];
		for(int i=0;i<nofo;i++){
				oe1[i] = (neto[i]-traino[i])*neto[i]*(1-neto[i])*neto[i];
		}
		
		
		//we will update the weight of hidden layer and output layer
		for(int i=0;i<nofo;i++){
			for(int j=0;j<nofh;j++){
				who[i][j] = who[i][j] - (eta*oe1[i]);
			}
		}
		
	    //now applying delta rule in the hidden and input layer
		// i am doubtful about this step
	    double[] eo2 = new double[nofh];
	    for(int i=0;i<nofh;i++){
			 eo2[i] =  (neto[i]-traino[i])*neth[i]*(1-neth[i])*neto[i];
		}
			 
		for(int i=0;i<nofh;i++){
		   for(int j=0;j<nofi;j++)	
			  wih[i][j] = wih[i][j] - (eta*eo2[i]);
		}
		//this is the final update we have got 
	}

}
