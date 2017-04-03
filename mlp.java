import java.io.*;
import java.util.*;

public class mlp {
	public static void main(String[] args) {
	System.out.print("The goal of backpropagation is to optimize the weights \nso that the neural network can learn how to \ncorrectly map arbitrary inputs to outputs.");
	Scanner s = new Scanner(System.in);
		
		System.out.print("Enter the value of no of input nodes ");
		int nofi = s.nextInt();
		
		System.out.print("Enter the value of no of hidden nodes ");
		int nofh = s.nextInt();
		
		System.out.print("Enter the value of no of output nodes ");
		int nofo = s.nextInt();
		
		double[][] wih = new double[nofi][nofh];
		double[][] who = new double[nofh][nofo];
		double[][] wihn = new double[nofi][nofh];
		double[][] whon = new double[nofh][nofo];
		double b1,b2; 
		double eta = 0.5;
		int epooch = 2500;
		b1 = s.nextDouble();
		b2 = s.nextDouble();
		
		System.out.print("Enter the value of weight of input nodes to hidden nodes");
		for(int i=0;i<nofi;i++) {
			for(int j=0;j<nofh;j++) {
				 wih[i][j] = s.nextDouble();
			}
		}
		
		System.out.print("Enter the value of weight of hidden nodes to output nodes connection");
		for(int i=0;i<nofh;i++) {
			for(int j=0;j<nofo;j++) {
				who[i][j] = s.nextDouble();
			}
		}
		
		double traini[] = new double[nofi];
		double traino[] = new double[nofo];
		
		System.out.print("Enter the training data of input");
		for(int i=0;i<nofi;i++) {
			traini[i] = s.nextDouble();
		}
		System.out.print("Enter the traning data of output");
		for(int i=0;i<nofo;i++) {
			traino[i] = s.nextDouble();
		}
		
		for(int no_of_times=0;no_of_times<epooch;no_of_times++) {
		    
			 
			       //forward pass 
		           double[] neth = new double[nofh];
		           for(int j=0;j<nofi;j++) {
		                    for(int i=0;i<nofh;i++) {
			                      neth[j] += wih[i][j]*traini[i];
		                    }
		                    neth[j] += b1*(1);
		                    //when u apply the logistic function to get the output of h1
		                    neth[j] = (1/( 1 + Math.pow(Math.E,(-1*neth[j]))));
		            }
		             
		         
		         // here we do the same for in order to get the output 
		         double[] neto = new double[nofo];
		         for(int j=0;j<nofo;j++) {
			            for(int i=0;i<nofh;i++) {
				                neto[j] += who[i][j]*neth[i];
			            }
			            neto[j] += b2*1;
			            neto[j] =  (1/( 1 + Math.pow(Math.E,(-1*neto[j]))));
		         }
		         
		         double[] error = new double[nofo];
		         double two= 2.00;
		         double totalerror = 0.00;
		
		       //calculating total error
		       for(int i=0;i<nofo;i++) {
			        error[i] = ((0.5) * (traino[i]-neto[i])*(traino[i]-neto[i]));
			        totalerror += error[i];
		       }
		   
		       //Backward pass
		      //Our goal with backpropagation is to update each of the weights in the network so that they cause the actual output to be closer the target output,
		      //thereby minimizing the error for each output neuron and the network as a whole.
		      
		       
		      //we have to see the weight change in hidden layer of the
		      for(int i=0;i<nofo;i++){
		    	  // calculating basic parameters for corresponding output value
		    	  double ecbyweight = 0.0;
		    	  for(int j=0;j<nofh;j++){
		    		  ecbyweight = (Math.abs(traino[i]-neto[i]))*neto[i]*(Math.abs(1-neto[i]))*neth[j];
		    		  whon[j][i] = who[j][i] - eta*ecbyweight;
		    	  }
		    	  
		      }
		      
		      for(int i=0;i<nofo;i++){
		    	  for(int j=0;j<nofh;j++){
		    		  System.out.print(whon[i][j]+"\n");
		    	  }
		      }
		      
		      for(int i=0;i<nofh;i++){
		    	  double ecbyweight = 0.0;
		    	  for(int j=0;j<nofi;j++){
		    		  ecbyweight = Math.abs(traini[i]-neth[i])*neth[i]*Math.abs(1-neth[i])*traini[j];
		    		  wihn[j][i] = wih[j][i] - eta*ecbyweight;
		    	  }
		      }
		      
		      for(int i=0;i<nofo;i++){
		    	  for(int j=0;j<nofh;j++){
		    		  System.out.print(wihn[i][j]+"\n");
		    	  }
		      }
		}
		
	}
}
 
