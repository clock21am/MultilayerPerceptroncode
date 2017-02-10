import java.io.*;
import java.util.*;
import java.util.Random;


public class tinypractice {
	/**
	 * @param args
	*/
	public static double[] nonlin(double[] syno,boolean deriv){
		if(deriv == true){
			return 
		}
		return syno;
	}
	
	public static double[] maxtrixmulti(double[][] in,double[] wi){
		double ans[] = new double[in.length]; 
		for(int i=0; i< in.length; i++){
			double c=0;
			for(int j=0;j<wi.length;j++){
				double l = wi[j];
				double m = in[i][j];
				c += l*m;
			}
			ans[i]=c;
		}
		return ans;
	}
	
	public static void main(String[] args) {

		double[][] in = { {0,0,1},
				      {0,1,1},
				      {1,0,1},
				      {1,1,1}
		};
		double[] out = {0,0,1,1};
		int maxEpoch = 100;
		//first layer of weights, synapse 0, connecting in to out
		double[] wi;
		wi = new double[4];
		for(int i=0;i<4;i++){
			wi[i] = 2*Math.random()-1;
		}
		
		double [][] fl;
		double [] sl;
		for(int epoch = 0; epoch < maxEpoch; epoch++){
			//forward propogation
			fl = in;
			//just to determine the matrix multiplication
		    sl = maxtrixmulti(fl,wi);
		    //applying sigmoid function 
			sl = nonlin(sl,false);
		}
	}

}
