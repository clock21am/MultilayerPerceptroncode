import java.io.*;
import java.util.*;
public class finalCode {
	public static void main(String[] args) {
		double max_epooch=100;
		
		// as mentioned in the example explained by the sir i have taken the 2 input nodes, 2 hidden nodes and 1 ouput node :)
		int input[] = {1,2};
		double wih[][] = { {0.5,0.6}, 
				           {0.5,0.4}
		};
		int hidden[] = {2,3};
		//it means we have two hidden nodes i don't know whether i should initialize some values in the node or not
		double wjh[] = {2.3,3.3};
		int output[] = {4};
		//outer loop for 100 test cases :p just say
		for(int epooch=1;epooch<max_epooch;epooch++) {
				
				// step 1---for calculating the sigmoid function for input and hidden
				double[] zn = new double[2];
				double oj = 0.0 ; // as we have only one node in the output we will store the 
				for(int i=0;i<2;i++){
					for(int j=0;j<2;j++){
						zn[i] += input[i]+ wih[j][i];
					}
				}
				
				// step 2---applying sigmoid function in the first input and hidden node one
				for(int i=0;i<2;i++){
					zn[i] = (1/( 1 + Math.pow(Math.E,(-1*zn[i]))));
				}
				
				// step 3----calculate weight with the output and hidden function
				double[] sn = new double[2];
				for(int i=0;i<2;i++){
					sn[i] = zn[i]*wjh[i];
					sn[i] = (1/( 1 + Math.pow(Math.E,(-1*zn[i]))));
					oj = oj + sn[i];
				}
				
				// summation of error ej = (oi - tj)2  do i need to take the ti value as manually or it will be calculated by some measures
				// d_wih and d_wjh 
				//for tj i guess output[0] will be there
				double ej;
				ej = oj - output[0];
				// now i followed by the gradient calculation step no 4
				
				double d_ih = 0.0;
				double d_hj = 0.0;
				
				// calculating the value of d_ih
				for(int i=0;i<2;i++){
					for(int j=0;j<2;j++){
						d_ih += -ej*zn[i]*wih[i][j]*sn[i]*input[i];
					}
				}
				//calculating the value of d_hj
				for(int i=0;i<2;i++){
					d_hj += -ej*sn[i]*zn[i];
							
				}
				
				for(int i=0;i<2;i++){
					for(int j=0;j<2;j++){
						 wih[i][j] = wih[i][j] + (d_ih/max_epooch);
					}
				}
				
				for(int j=0;j<2;j++){
					wjh[j] = wjh[j] + (d_hj/max_epooch);
				}
		}
	}

}
