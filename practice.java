import java.io.*;
import java.util.*;
public class practice {
	
	public static double[] nonlin(double[] syno,boolean deriv){
		double ans[] = new double[syno.length];
		if(deriv == true){
		    //complement of the syno for returning x*(1-x)
			for(int i=syno.length-1,j=0;i>=0;i--){
				if(syno[i]==0){
					ans[j]=1;
					j++;
				}else {
					ans[j]=0;
					j++;
				}
			}
			for(int i=0;i<syno.length;i++){
				ans[i]=ans[i]*syno[i];
			}
		}else {
			//calculating exp(-x)
			for(int i=0;i<syno.length;i++){
				ans[i]=ans[i]*(-1);
				ans[i]=Math.exp(ans[i]);
				ans[i]=ans[i]+1;
			}
			for(int i=0;i<syno.length;i++){
				ans[i]=1/(ans[i]);
			}
		}
		return ans;
	}
	
	public static double[] maxtrixmulti(double[][] in,double[] wi){
		double ans[] = new double[in.length]; 
		for(int i=0; i< in.length-1; i++){
			double c=0;
			for(int j=0;j<wi.length-1;j++){
				double l = wi[j];
				double m;
				if(j<in.length){
				    m = in[i][j];
				}else {
				    m = 0;
				}
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
				          {1,1,1}, 
		               };
		double[] out = {0,0,1,1};
		int maxEpoch = 10;
		//first layer of weights, synapse 0, connecting in to out
		double[] wi;
		wi = new double[4];
		for(int i=0;i<4;i++){
			wi[i] = 2*Math.random()-1;
		}
		
		double [][] fl;
		double [] sl;
		double [] l1_error;
		l1_error = new double[out.length];
		double [] l1_delta = new double[out.length];
		
		for(int epoch = 0; epoch < maxEpoch; epoch++) { 
			
			//forward propogation
			fl = in;
			
			//just to determine the matrix multiplication so we can apply the sigmod function
		    sl = maxtrixmulti(fl,wi);
		    
		    //applying sigmoid function 
			sl = nonlin(sl,false);
			
			for(int i=0;i<sl.length;i++){
				System.out.println(sl[i]+" ");
			}
			System.out.println("\n");
			//print till forward propogation
			
			//how much we miss ? ? 
			for(int i=0;i<out.length;i++) {
				l1_error[i] = out[i] - sl[i];
			}
			
			//applying sigmoid at the value in l1
			sl=nonlin(sl,true);
			
			//multiply by the value that how much we have missed
			for(int i=0;i<out.length;i++){
				l1_delta[i] = l1_error[i]*sl[i]; 
			}
			
			//dot product of the input clone and l1_delta
			l1_delta = maxtrixmulti(fl,l1_delta);
			
			//update weight product
			
			for(int i=0;i<out.length;i++){
				sl[i]=sl[i]+l1_delta[i];
			}
			
			for(int i=0;i<out.length;i++){
				System.out.println(sl[i]+" ");
			}
			System.out.println("\n");
		}   
	}
}
