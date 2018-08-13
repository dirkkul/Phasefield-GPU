#include "PF_GPU.hu"

__device__ __constant__ ParD d_par; //Parameterstruct on GPU

texture<float2,2,cudaReadModeElementType> tex_SM_DiffBend;

__global__ void d_DiffusionFFT(dPointer d_point){
	unsigned int const x=threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int const y=threadIdx.y + blockIdx.y * blockDim.y;
	
	int const offset = x + y * d_par.N;
	
	if(offset < d_par.N2){
		float const coeff = d_point.SpecMethDiffBend[offset];
		for(int CellIdx = 0; CellIdx < d_par.CN/2; CellIdx++){
			d_point.CellsOut[offset + d_par.N2 * CellIdx] = coeff * d_point.CellsIn[offset + d_par.N2 * CellIdx];
		}
	}
};

__global__ void d_PreparePhiGDeriv(dPointer d_point){
	unsigned int const x=threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int const y=threadIdx.y + blockIdx.y * blockDim.y;
	int const offset = x + y * d_par.N;	
	if(offset < d_par.N2){
		float Phi;
		for(int CellIdx = 0; CellIdx < d_par.CN;  CellIdx++){
			
			if(CellIdx%2 == 0){
				Phi = d_point.CellsIn[offset + CellIdx/2*d_par.N2].x;
				d_point.FftIn[offset + CellIdx*d_par.N2].x = Phi;
				d_point.FftIn[offset + (CellIdx + 1) * d_par.N2].x = 36.0 * Phi * (1.0 - Phi)*(1.0 - 2.0 * Phi);
			}else{
				Phi = d_point.CellsIn[offset + (CellIdx-1)/2*d_par.N2].y;
				d_point.FftIn[offset + (CellIdx - 1) * d_par.N2].y = Phi;
				d_point.FftIn[offset + CellIdx*d_par.N2].y = 36.0 * Phi * (1.0 - Phi)*(1.0 - 2.0 * Phi);
			}
			
			d_point.phiOld[offset + CellIdx*d_par.N2] = Phi;
			
		}
	}
}


__global__ void d_PhiGDeriv(dPointer d_point){
	unsigned int const x=threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int const y=threadIdx.y + blockIdx.y * blockDim.y;
	int const offset = x + y * d_par.N;	
	if(offset < d_par.N2){
		float4 const coeff = d_point.SpectralGradLap[offset];
		
		for(int Cell = 0; Cell < d_par.CN;  Cell+= 2){
			d_point.FftOut[offset + 4 * Cell/2 * d_par.N2] 		= d_point.FftIn_w[offset + Cell 		* d_par.N2] * coeff.x;
			d_point.FftOut[offset + (4 * Cell/2 + 1) * d_par.N2]	= d_point.FftIn_w[offset + (Cell + 1) * d_par.N2] * coeff.x;
			
			float2 InVar = d_point.FftIn_w[offset + Cell	* d_par.N2];
			float2 OutGradX, OutGradY;
			OutGradX.y	= InVar.x * coeff.z;
			OutGradY.y	= InVar.x * coeff.w;
			OutGradX.x = -InVar.y * coeff.z;
			OutGradY.x = -InVar.y * coeff.w;
			
			d_point.FftOut[offset + (4 * Cell/2 + 2) * d_par.N2] = OutGradX;
			d_point.FftOut[offset + (4 * Cell/2 + 3) * d_par.N2] = OutGradY;
		}
	}
}

__global__ void d_PhiExpl(dPointer d_point){
	unsigned int const x=threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int const y=threadIdx.y + blockIdx.y * blockDim.y;
	
	int const offset = x + y * d_par.N;
	if(offset < d_par.N2){
		float DivX, DivY, laplace, PotGrad, Phi;
		
		for(int CellIdx = 0; CellIdx < d_par.CN; CellIdx++){
			float const Rho = d_point.RDIn[offset+CellIdx*d_par.N2].x;
			if(CellIdx%2 == 0){
					Phi = d_point.CellsIn[offset + d_par.N2 * CellIdx].x ;
					laplace = d_point.FftOut_w[offset + 4*CellIdx/2 	* d_par.N2].x;
					PotGrad = d_point.FftOut_w[offset + (4*CellIdx/2+1) * d_par.N2].x;

					DivX = d_point.FftOut_w[offset + (4*CellIdx/2+2) * d_par.N2].x; 
					DivY = d_point.FftOut_w[offset + (4*CellIdx/2+3) * d_par.N2].x;
			}else{
					int const CellField = (CellIdx - 1)/2;
					Phi = d_point.CellsIn[offset + d_par.N2 * CellField].y;
					laplace = d_point.FftOut_w[offset + 4*CellField/2 	* d_par.N2].y;
					PotGrad = d_point.FftOut_w[offset + (4*CellField/2+1) * d_par.N2].y;
					
					DivX = d_point.FftOut_w[offset + (4*CellField+2) * d_par.N2].y;
					DivY = d_point.FftOut_w[offset + (4*CellField+3) * d_par.N2].y;
			}
			float gradAbs = DivX * DivX + DivY * DivY;
			
			float2 GDiff;
			GDiff.x = 36.0 * Phi * (1.0 - Phi)*(1.0 - 2.0 * Phi);
			GDiff.y = 36.0 * (1.0 - 6.0 * Phi + 6.0*Phi*Phi);
			
			float const PhiUpdate = Phi
			+ d_par.kappa * (PotGrad + GDiff.y*(laplace-GDiff.x/d_par.epsilon2))
			- d_par.gamma * GDiff.x
			+ (d_par.alpha*Rho - d_par.beta) * sqrt(gradAbs);
			
			
			//~ - d_par.gamma * GDiff.x
			//~ + (d_par.alpha*Rho - d_par.beta);
			
			if(CellIdx%2 == 0){
				d_point.CellsOut[offset + d_par.N2 * CellIdx/2].x = PhiUpdate;
			}else{
				d_point.CellsOut[offset + d_par.N2 * (CellIdx - 1)/2].y = PhiUpdate;
			}
			//~ - par->grep * (repul[indBig].x - Phi)
			//~ - par->sigma * AdhGrad;
		}
	}
}

__global__ void d_UpdateRD(dPointer d_point){
	unsigned int const x=threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int const y=threadIdx.y + blockIdx.y * blockDim.y;
	
	int const offset = x + y * d_par.N;
	if(offset < d_par.N2){
		int Right = offset + 1;
		if(x == d_par.N - 1) Right = offset + 1 - d_par.N;
		int Left = offset - 1;
		if(x==0) Left = offset - 1 + d_par.N;
		int Front = offset + d_par.N; 
		if(y == d_par.N - 1)Front = offset + d_par.N - d_par.N2;
		int Back = offset - d_par.N;
		if(y == 0) Back = offset - d_par.N + d_par.N2;
		
		float2 reak, RDdiv;
		for(int CellIdx = 0; CellIdx < d_par.CN; CellIdx++){
			
			int const CellInd = CellIdx * d_par.N2;
			
			float2 const RD = d_point.RDIn[offset + CellInd];
			
			float PhiNew;
			if(CellIdx %2 == 0){
				PhiNew = d_point.CellsIn[offset + CellIdx/2 * d_par.N2].x;
			}else{
				PhiNew = d_point.CellsIn[offset + (CellIdx-1)/2 * d_par.N2].y;
			}
			float const Phi = d_point.phiOld[offset + CellInd];
			
			float const rho2 = RD.x*RD.x;
			reak.x = d_par.k_b*(rho2/(d_par.KK_a + rho2)+d_par.k_a)*d_point.RhoTot[CellIdx] - d_par.k_c*(1+0.)*RD.x;
			reak.y = - d_par.k_Ib * RD.y;
			
			float const normal = d_par.eta * 0.;
			
			float2 const ReakDiff =  (2*Phi - PhiNew) * RD + Phi * reak + d_par.DiffRD * (
																			  (Phi + d_point.phiOld[Right + CellInd]) * (d_point.RDIn[Right + CellInd] - RD) 
																			- (Phi + d_point.phiOld[Left + CellInd]) * (RD - d_point.RDIn[Left + CellInd]) 
																			+ (Phi + d_point.phiOld[Front + CellInd]) * (d_point.RDIn[Front + CellInd] - RD) 
																			- (Phi + d_point.phiOld[Back + CellInd]) * (RD - d_point.RDIn[Back + CellInd]) );
			
			if(Phi >= 0.0001){
				RDdiv.y = ReakDiff.y/Phi + normal;
				RDdiv.x = ReakDiff.x/Phi;
			}else{
				RDdiv.y = ReakDiff.y + normal * Phi;
				RDdiv.x = ReakDiff.x;

			}
			
			d_point.test[offset + CellInd] = d_par.k_c*RD.x;
			d_point.RDOut[offset + CellInd] = RDdiv;
		}
	}
}

__global__ void d_Position(dPointer d_point, int run){
	int offset0		= threadIdx.x;
	int cacheidx	= threadIdx.x;
	int CellIdx		= blockIdx.x;
	__shared__ float Ax1cache[MAXT], Ax2cache[MAXT], Ay1cache[MAXT], Ay2cache[MAXT];
	
	int x, y, offset;
	float PF, Ax1, Ax2, Ay1, Ay2;
	
	if(CellIdx < d_par.CN){
		while(offset0 < d_par.N){
			offset = offset0;
			while(offset < d_par.N2){
				x= offset%d_par.N;
				y= offset/d_par.N;
				
			if(CellIdx %2 == 0)
				PF = d_point.CellsIn[offset + CellIdx/2 * d_par.N2].x;
			else
				PF = d_point.CellsIn[offset + (CellIdx-1)/2 * d_par.N2].y;
				
				Ax1 += d_point.ComPos[x].x * PF;
				Ax2 += d_point.ComPos[x].y * PF;
				Ay1 += d_point.ComPos[y].x * PF;
				Ay2 += d_point.ComPos[y].y * PF;
				offset += d_par.N;
			}
			offset0 += blockDim.x;
		}
		
		Ax1cache[cacheidx]	= Ax1;
		Ax2cache[cacheidx]	= Ax2;
		Ay1cache[cacheidx]	= Ay1;
		Ay2cache[cacheidx]	= Ay2;
		
		__syncthreads();
		
		for(unsigned int s = blockDim.x/2; s>0; s>>=1){
			if(cacheidx < s){
				Ax1cache[cacheidx] 	+= Ax1cache[cacheidx + s];
				Ax2cache[cacheidx] 	+= Ax2cache[cacheidx + s];
				Ay1cache[cacheidx] 	+= Ay1cache[cacheidx + s];
				Ay2cache[cacheidx] 	+= Ay2cache[cacheidx + s];
			}
			__syncthreads();
		}
		
		if(cacheidx == 0){
			d_point.Pos[CellIdx*d_par.NumSave + run].x = (atan2(-Ax2cache[0], -Ax1cache[0]) + M_PI)/d_par.DeltaX;
			d_point.Pos[CellIdx*d_par.NumSave + run].y = (atan2(-Ay2cache[0], -Ay1cache[0]) + M_PI)/d_par.DeltaX;
		}
	}
};

__global__ void d_SumCont(dPointer d_point){
	unsigned int const x=threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int const y=threadIdx.y + blockIdx.y * blockDim.y;
	
	int const offset = x + y * d_par.N;
	if(offset < d_par.N2){
		float4 Sum = make_float4(0, 0, 0, 0);
		float2 RD;
		float  Fields;
		for(int CellIdx = 0; CellIdx < d_par.CN; CellIdx++){
			if(CellIdx %2 == 0)
				Fields = d_point.CellsIn[offset + CellIdx/2 * d_par.N2].x;
			else
				Fields = d_point.CellsIn[offset + (CellIdx-1)/2 * d_par.N2].y;
			RD = d_point.RDOut[offset + CellIdx*d_par.N2];
			Sum.x += Fields;
			//~ Sum.y += ;
			Sum.z += RD.x*Fields;
			Sum.w += RD.y;
		}
		d_point.Sum[offset] = Sum;
	}
	
};

__global__ void d_SumRho(dPointer d_point){
	int offset0		= threadIdx.x;
	int cacheidx	= threadIdx.x;
	int CellIdx		= blockIdx.x;
	
	__shared__ float RDcache[MAXT], PhiCache[MAXT];
	float Rho = 0, PF = 0, Phi;
	
	if(CellIdx < d_par.CN){
		while(offset0 < d_par.N){
			int offset = offset0;
			while(offset < d_par.N2){
				if(CellIdx %2 == 0)
					Phi = d_point.CellsIn[offset + CellIdx/2 * d_par.N2].x;
				else
					Phi = d_point.CellsIn[offset + (CellIdx-1)/2 * d_par.N2].y;
				PF += Phi;
				Rho += Phi*d_point.RDIn[offset + CellIdx*d_par.N2].x;
				offset += d_par.N;
			}
			offset0 += blockDim.x;
		}
	
		PhiCache[cacheidx] = PF;
		RDcache[cacheidx] = Rho;
		
		__syncthreads();
		
		for(unsigned int s = blockDim.x/2; s>0; s>>=1){
			if(cacheidx < s){
				RDcache[cacheidx] += RDcache[cacheidx + s];
				PhiCache[cacheidx] += PhiCache[cacheidx + s];
			}
			__syncthreads();
		}
		
		if(cacheidx == 0){
			d_point.RhoTot[CellIdx] = (d_par.rhotot - RDcache[cacheidx]* d_par.dx2)/(PhiCache[cacheidx]  * d_par.dx2);
		}
		
	}
}

void d_FFT_TimeStep(dPointer d_point, cufftHandle &fftPlan, bool fwd=true){
	if(fwd){
		cufftExecC2C(fftPlan,(cufftComplex*) d_point.CellsOut,(cufftComplex*) d_point.CellsIn,CUFFT_FORWARD);
	}else{
		cufftExecC2C(fftPlan,(cufftComplex*) d_point.CellsOut,(cufftComplex*) d_point.CellsIn,CUFFT_INVERSE);
	}
};

int main(int argc, char** argv){
	if( argc != 2){
		std::cout<< "Only give the path as argument"<<std::endl;
		exit(1);
	}
	
	//ParameterStruct
	ParD par;
	ParHost parHost;
	dPointer d_point;
	parHost.path = argv[1];
	
	ReadParamFromFile(&par, &parHost);
	Prepare(&par, &parHost);
	Scaling(&par, &parHost);
	
	cufftHandle BatchFFT_TimeStep, BatchFFT_DerivFor, BatchFFT_DerivBack;
	int n[2]={par.N,par.N};
	cufftPlanMany(&BatchFFT_TimeStep, 2, n, NULL, 1, par.N2, NULL, 1, par.N2, CUFFT_C2C, parHost.NumberCellFields);
	cufftPlanMany(&BatchFFT_DerivFor, 2, n, NULL, 1, par.N2, NULL, 1, par.N2, CUFFT_C2C, parHost.NumberCellFields*2);
	cufftPlanMany(&BatchFFT_DerivBack, 2, n, NULL, 1, par.N2, NULL, 1, par.N2, CUFFT_C2C, parHost.NumberCellFields*4);
	
	PrepareGPU(&par, &parHost);
	Assert(&par, &parHost);
	CudaDeviceMem(&par, &parHost, &d_point);
	SetUpPosM(&d_point, &par, &parHost);
	SetUpSpecM(&d_point, &par, &parHost);
	
	InitialConditions(&par, &parHost, &d_point);
	CudaSafeCall( cudaMemcpyToSymbol(d_par, &par, sizeof(ParD)) );

	//~ d_SumCont <<<parHost.blocks, parHost.threads>>>(d_point);
	//~ CudaCheckError();
	
	//~ PlotStates(&par, &parHost, &d_point, 0);
	
	//~ d_SumRho<<<parHost.blocks1D, parHost.threads1D>>>(d_point);
	//~ CudaCheckError();
	//~ d_UpdateRD <<<parHost.blocks, parHost.threads>>>(d_point);
	//~ CudaCheckError();
	//~ PlotRandomField(&par, &parHost, &d_point, 0);

	
	for(size_t t=0; t<parHost.EndSteps+1; t++){
		
		
		d_SumRho<<<parHost.blocks1D, parHost.threads1D>>>(d_point);
		CudaCheckError();
		
		//~ float *RhoTot  = new float[par.CN];
		//~ CudaSafeCall( cudaMemcpy(RhoTot , d_point.RhoTot, par.CN* sizeof(float), cudaMemcpyDeviceToHost) );
		//~ std::cout <<RhoTot[0]<< " "<<RhoTot[1] << std::endl;
		//~ delete[] RhoTot;

		d_PreparePhiGDeriv<<<parHost.blocks, parHost.threads>>>(d_point);
		CudaCheckError();
		
		cufftExecC2C(BatchFFT_DerivFor,(cufftComplex*) d_point.FftIn,(cufftComplex*) d_point.FftIn_w,CUFFT_FORWARD);
		CudaCheckError();
		
		d_PhiGDeriv<<<parHost.blocks, parHost.threads>>>(d_point);
		CudaCheckError();
		
		cufftExecC2C(BatchFFT_DerivBack,(cufftComplex*) d_point.FftOut,(cufftComplex*) d_point.FftOut_w,CUFFT_INVERSE);
		CudaCheckError();
		
		//~ float2 *FftOut_w  = new float2[par.N2*4*parHost.NumberCellFields];
		//~ CudaSafeCall( cudaMemcpy(FftOut_w , d_point.FftOut_w, par.N2 *parHost.NumberCellFields*4* sizeof(float2), cudaMemcpyDeviceToHost) );
		//~ std::cout <<FftOut_w[0].x << std::endl;
		//~ exit(1);
		
		d_PhiExpl<<<parHost.blocks, parHost.threads>>>(d_point);
		CudaCheckError();
		
		
		
		d_FFT_TimeStep(d_point, BatchFFT_TimeStep);
		CudaCheckError();
		
		d_DiffusionFFT <<<parHost.blocks, parHost.threads>>>(d_point);
		CudaCheckError();
		
		d_FFT_TimeStep(d_point, BatchFFT_TimeStep,false);
		CudaCheckError();
		
		d_UpdateRD <<<parHost.blocks, parHost.threads>>>(d_point);
		CudaCheckError();
		
		if(t%parHost.SaveSteps==0){//Save computed data
			int run=int(t/parHost.SaveSteps);
			std::cout <<"run: " << run << std::endl;
			d_Position<<<parHost.blocks1D, parHost.threads1D>>>(d_point, run);
			CudaCheckError();
			
			d_SumCont <<<parHost.blocks, parHost.threads>>>(d_point);
			CudaCheckError();
			
			PlotStates(&par, &parHost, &d_point, run);
			PlotRandomField(&par, &parHost, &d_point, run);
		}
		
		swap(d_point.RDIn, d_point.RDOut);
	}
	
	WritePosition(&par, &parHost, &d_point);
	
	CudaDeviceMemFree(&d_point);
	
	cufftDestroy(BatchFFT_DerivFor);
	cufftDestroy(BatchFFT_TimeStep);
}

/* \fn ChooseGPU()
 *	\brief chooses the gpu card with most multiprocessors for cuda calculations
 */
void ChooseGPU(){
	//choose the right card, we choose the card with most multiProcessors
	int num_devices, device;
	cudaGetDeviceCount(&num_devices);
	
	//only do this, when we have more than one device
	int max_device = 0;
	if (num_devices > 0) {
		int max_multiprocessors = 0;
		for (device = 0; device < num_devices; device++) {
				cudaDeviceProp properties;
				cudaGetDeviceProperties(&properties, device);
				
				//check number of multiProcessor and store the number and the index
				if (max_multiprocessors < properties.multiProcessorCount) {
						max_multiprocessors = properties.multiProcessorCount;
						max_device = device;
				}
		}
		cudaSetDevice(max_device);
	}else{
		std::cout << "no cuda device found, exiting";
		exit(1);
	}
}

void Assert(ParD* par, ParHost* parHost){
	assert(par->N > 0);
	assert(par->L > 0);
	assert(par->NumSave > 0);
	assert(parHost->SaveSteps > 0);
	assert(parHost->EndSteps > 0);
	assert(parHost->EndSteps > parHost->SaveSteps);
}

/* \fn ComputeDistanceDimension(ParamW* , size_t, ---)
 * \brief Computes the Distance between two Points for all Dimension with respect to periodic Boundary conditions
 * \param paramW: Struct in which we store all values. This struct will be copied to the constant device Memory later
 * \param x1: Index of the Point
 */
float ComputeDistance(int size, int x1, int x2, int y1, int y2){
	return sqrt(pow(ComputeDistanceDimension(x1,x2, 2, size),2) + pow(ComputeDistanceDimension(y1,y2, 2, size),2));
}

/* \fn ComputeDistanceDimension(size_t, size_t, size_t, size_t)
 * \brief Computes the Distance for two Points in one Dimension with respect to periodic Boundary conditions
 * \param Point1: Index of the First Point
 * \param Point2: Index of the second Point
 * \param Boundary: Index of the Boundary for the given Dimension
 * \param size: Size of the Dimension
 */
int ComputeDistanceDimension(int Point1, int Point2, int Boundary, int size){
	int Dist = Point1 - Point2;
	
	//periodic
	if(Boundary == 2){
		if(Point1 >size/2){   //upper boundary
			if(abs(Dist)> size/2){
				Dist = Point1 - ( Point2 + size);
			}
		}else{
			if(abs(Dist)> size/2){
				Dist = size - abs(Dist);
			}
		}
	}
	
	return Dist;
}

void CudaDeviceMem(ParD* par, ParHost* parHost, dPointer * d_point ){
	CudaSafeCall(cudaMalloc(&(d_point->Pos)					, par->CN * par->NumSave * sizeof(float2)) );
	CudaSafeCall(cudaMalloc(&(d_point->ComPos)				, par->N2 * sizeof(float2)) );
	CudaSafeCall(cudaMalloc(&(d_point->Sum)					, par->N2 * sizeof(float4)) );
	CudaSafeCall(cudaMalloc(&(d_point->SpecMethDiffBend)	, par->N2 * sizeof(float)) );
	CudaSafeCall(cudaMalloc(&(d_point->SpectralGradLap)		, par->N2 * sizeof(float4)) );
	CudaSafeCall(cudaMalloc(&(d_point->CellsIn)				, parHost->NumberCellFields * par->N2 * sizeof(float2)) );
	CudaSafeCall(cudaMalloc(&(d_point->CellsOut)			, parHost->NumberCellFields * par->N2 * sizeof(float2)) );
	CudaSafeCall(cudaMalloc(&(d_point->phiOld)				, par->CN * par->N2 * sizeof(float)) );
	CudaSafeCall(cudaMalloc(&(d_point->RDIn)				, par->CN * par->N2 * sizeof(float2)) );
	CudaSafeCall(cudaMalloc(&(d_point->RDOut)				, par->CN * par->N2 * sizeof(float2)) );
	CudaSafeCall(cudaMalloc(&(d_point->RhoTot)				, par->CN * sizeof(float)) );
	
	CudaSafeCall(cudaMalloc(&(d_point->test)				, par->CN  * par->N2* sizeof(float)) );
	
	CudaSafeCall(cudaMalloc(&(d_point->FftIn)				, parHost->NumberCellFields * par->N2 * 2* sizeof(float2)));
	CudaSafeCall(cudaMalloc(&(d_point->FftIn_w)				, parHost->NumberCellFields * par->N2 * 2* sizeof(float2)));
	CudaSafeCall(cudaMalloc(&(d_point->FftOut)				, parHost->NumberCellFields * par->N2 * 4* sizeof(float2)));
	CudaSafeCall(cudaMalloc(&(d_point->FftOut_w)			, parHost->NumberCellFields * par->N2 * 4* sizeof(float2)));
	
	//~ CudaSafeCall( cudaMemset(d_point->CellsOut, 0, parHost->NumberCellFields * par->N2 * sizeof(float2)) );
	//~ CudaSafeCall( cudaMemset(d_point->FftOut, 0, parHost->NumberCellFields * 4 *par->N2 * sizeof(float2)) );
	
	//~ cudaChannelFormatDesc descREAL2 = cudaCreateChannelDesc<float2>();
	//~ CudaSafeCall( cudaBindTexture2D( NULL, tex_SM_DiffBend , d_point->SpecMethDiffBend, descREAL2, par->N, par->N, par->N * sizeof(float2) ) );

}

void CudaDeviceMemFree(dPointer * d_point ){
	CudaSafeCall(cudaFree(d_point->SpecMethDiffBend) );
	CudaSafeCall(cudaFree(d_point->CellsIn) );
	CudaSafeCall(cudaFree(d_point->CellsOut) );
}

void InitialConditions(ParD* par, ParHost* parHost, dPointer * d_point){
	float2 *Cells	= new float2[parHost->NumberCellFields * par->N2]();
	float2 *RDIn	= new float2[par->CN * par->N2]();
	float2 *Pos		= new float2[par->CN];
	float *Direction	= new float[par->CN];
	
	InitializeStartDirection(par, parHost, d_point, Direction, Pos);
	//Fill the Cellfields with Data.
	
	for(size_t y=0; y<par->N; y++){
		for(size_t x=0; x<par->N; x++){
			size_t offset = x + par->N * y;
			
			for(int CellIdx=0; CellIdx < par->CN; CellIdx++){
				float distance = ComputeDistance(par->N,x, Pos[CellIdx].x, y, Pos[CellIdx].y)*par->dx;
				float Phi0 =  0.5+0.5*tanh(3*(parHost->R - distance)/parHost->epsilon);
				if(CellIdx %2 == 0)
					Cells[offset + CellIdx/2 * par->N2].x = Phi0;
				else
					Cells[offset + (CellIdx-1)/2 * par->N2].y = Phi0;
				
				float arcDist  = fabs(atan2(1.0 * (y - Pos[CellIdx].y),1.0 * (x - Pos[CellIdx].x) ) + M_PI - Direction[CellIdx] );
				
				if(arcDist <0) arcDist += 2*M_PI;
				if(1.0 * M_PI/2.0 > arcDist || arcDist > 3.0 * M_PI/2.0 ){
					//~ RDIn[offset + par->N2*CellIdx].x = (1.8 + rn())* Phi0;
					RDIn[offset + par->N2*CellIdx].x = (2.5)* Phi0;
				} 
			}
		}
	}
	
	CudaSafeCall( cudaMemcpy(d_point->CellsIn , Cells, parHost->NumberCellFields * par->N2 * sizeof(float2), cudaMemcpyHostToDevice) );
	CudaSafeCall( cudaMemcpy(d_point->CellsOut, Cells, parHost->NumberCellFields * par->N2 * sizeof(float2), cudaMemcpyHostToDevice) );
	CudaSafeCall( cudaMemcpy(d_point->RDIn, 	RDIn, par->CN * par->N2 * sizeof(float2), cudaMemcpyHostToDevice) );
	CudaSafeCall( cudaMemcpy(d_point->RDOut, 	RDIn, par->CN * par->N2 * sizeof(float2), cudaMemcpyHostToDevice) );
	
	std::string In = parHost->path+"/Inital_RD.dat";
	std::ofstream outInitial;
	outInitial.open (In.c_str(),std::ios::out );

	if (outInitial.is_open() ){
		for(int y=0; y< par->N; y++){
			for(int x=0; x< par->N; x++){
				float Sum =0;
				int offset = x + y * par->N;
				
				for(int CellIdx=0; CellIdx < par->CN; CellIdx++){
					Sum += RDIn[offset + par->N2*CellIdx].x;
				}
				outInitial	<< Sum << " ";
			}
			outInitial <<std::endl;
		}
	}
	outInitial.close();
	
	delete[] Cells;
	delete[] RDIn;
	delete[] Pos;
	delete[] Direction;
	
	
	
}

void InitializeStartDirection(ParD* par, ParHost* parHost, dPointer * d_point, float* AngleVec, float2* Position){
	//Calculate the maximal number of starting positions 
	//distance between the middle of two Cells should be two times the radius + some extra space. We also need to know how many Cells can fit in one line.
	int dist = 2 * parHost->R/par->dx + 15;
	int oneLine = max(par->N/ dist, 1);
	
	int NumberCellPositions = oneLine * oneLine;
	
	//exit if there are more Cells than startpositions
	if(par->CN>NumberCellPositions){
		std::cout << "More Cells than possible Startpositions, exiting"<<std::endl;
		exit(1);
	}
	
	//create all possible start positions
	std::vector <int2> PossibleStartPositions(NumberCellPositions);
	for(int i = 0; i < NumberCellPositions; i++){
		PossibleStartPositions[i].x = (i % oneLine) * 2* dist + parHost->R/par->dx * 1.5;
		PossibleStartPositions[i].y = (i / oneLine) * 2 *dist + parHost->R/par->dx * 1.5;
	}
	
	//create a random list from 0 to cellnumber. Shuffle the list so cells picks randomly one of the Positions
	std::vector<int> StartList(NumberCellPositions);
	generate (StartList.begin(), StartList.end(), UniqueNumber);
	std::random_shuffle( StartList.begin(), StartList.end() );
	
	
	//choose from all possible positions
	std::vector <int2> StartPositions(par->CN);
	for(int CellIdx = 0; CellIdx < par->CN; CellIdx++){
		Position[CellIdx].x = fmod(PossibleStartPositions[StartList[CellIdx]].x + parHost->R/par->dx * rn() + par->N,par->N);
		Position[CellIdx].y = fmod(PossibleStartPositions[StartList[CellIdx]].y + parHost->R/par->dx * rn() + par->N,par->N);
	}
	
	CudaSafeCall( cudaMemcpyAsync(d_point->Pos,	Position, par->CN * sizeof(float2), cudaMemcpyHostToDevice) );
	
	//random start Directions
	for(int CellIdx = 0; CellIdx < par->CN; CellIdx++){
		AngleVec[CellIdx] = 2.0*M_PI*drand48();
	}
	
	//print Start positions
	for(int CellIdx = 0; CellIdx < par->CN; CellIdx++){
		std::cout << "StartPosition for Cell: "<<  CellIdx << " x: "<< Position[CellIdx].x << " physical pos:" <<Position[CellIdx].x * par->dx <<" y: " << Position[CellIdx].y<< " physical pos:" <<Position[CellIdx].y * par->dx << ". With Direction: " << AngleVec[CellIdx] << std::endl;
	}
	
	//write the startpositions to file
	std::string file=parHost->path+"StartPositions.dat"; //add path to file
	
	std::ofstream StartPosFile;
	StartPosFile.open (file.c_str(),std::ios::out );	//open file
	if (StartPosFile.is_open()){
		for(int CellIdx = 0; CellIdx < par->CN; CellIdx++){
			StartPosFile << "[Cell"<< CellIdx <<"]" <<std::endl
			<< "StartPositionX=" << Position[CellIdx].x 
			<< "\nStartPositionY=" << Position[CellIdx].y 
			<< "\nStartAngle=" << AngleVec[CellIdx] << std::endl;
		}
	}else{
		std::cout << "Unable to write to StartPositions.dat file";
		exit(1);
	}
	StartPosFile.close();
}

void PlotStates(ParD* par, ParHost* parHost, dPointer *d_point, int run){
	if(not parHost->PlotStates) return;
	
	float4 *Sum  = new float4[par->N2];
	CudaSafeCall( cudaMemcpy(Sum , d_point->Sum , par->N2 * sizeof(float4), cudaMemcpyDeviceToHost) );
	
	char RunChar[10];
	sprintf(RunChar, "%5.5d", run);
	std::string PfName = "PhaseField";
	std::string PName = "Polarisation";
	std::string fileP= parHost->path+"/"+PName+"_"+RunChar+".dat";
	std::string filePF	= parHost->path+"/"+PfName+"_"+RunChar+".dat";
	std::ofstream outP, outPF;
	outP.open (fileP.c_str(),std::ios::out );
	outPF.open (filePF.c_str(),std::ios::out );

	if (outP.is_open() && outPF.is_open()){
		for(int y=0; y< par->N; y++){
			for(int x=0; x< par->N; x++){
				int offset = x + y * par->N;
				outPF	<< Sum[offset].x << " ";
				outP 	<< Sum[offset].z << " ";
			}
			outPF <<std::endl;
			outP <<std::endl;
		}
	}
	outPF.close();
	outP.close();
	
	delete[] Sum;
	
	//~ std::string cmd0 = "python plotCombinedData.py "+parHost->path+" &";
	std::string cmd0 = "python plotCombinedData.py "+parHost->path;
	system(cmd0.c_str());
	//~ sleep(2);
}

void PlotRandomField(ParD* par, ParHost* parHost, dPointer *d_point, int run){
	if(not parHost->PlotStates) return;
	
	float *F  = new float[par->N2];
	CudaSafeCall( cudaMemcpy(F , d_point->test, par->N2 *sizeof(float), cudaMemcpyDeviceToHost) );
	
	char RunChar[10];
	sprintf(RunChar, "%5.5d", run);
	std::string Name = "Field";
	std::string file	= parHost->path+"/"+Name+"_"+RunChar+".dat";
	std::ofstream out;
	out.open (file.c_str(),std::ios::out );

	for(int y=0; y< par->N; y++){
		for(int x=0; x< par->N; x++){
			int offset = x + y * par->N;
			out	<< F[offset] << " ";
		}
		out <<std::endl;
	}
	out.close();
	
	delete[] F;
	
	std::string cmd0 = "python plotField.py "+parHost->path;
	system(cmd0.c_str());
}

void Prepare(ParD* par, ParHost* parHost){
	//seed the prng. if it has an negative value, use the time as seed
	if(parHost->seed <0){	parHost->seed = time( NULL );  }
	srand48( (unsigned int) parHost->seed );
	srand ( (unsigned int)  parHost->seed);
	
	parHost->NumberCellFields = (par->CN+1)/2;
	par->N2 = par->N*par->N;
	par->dx = par->L/par->N;
	par->dx2 = par->dx*par->dx;
	std::cout << par->dx << std::endl;
	std::cout << par->dx2 << std::endl;
	
	par->epsilon2 = parHost->epsilon*parHost->epsilon;
	parHost->dk = 2. * M_PI/par->L;
	
	parHost->SaveSteps = parHost->SaveTime/par->dt;
	parHost->EndSteps = parHost->EndTime/par->dt;
	
	par->NumSave = parHost->EndSteps/parHost->SaveSteps; //how often we save
}

void PrepareGPU(ParD* par, ParHost* parHost){
	//GPU Setup
	cudaDeviceReset();
	ChooseGPU();//chose GPU with most prozessors
	
	//How many blocks/threads are called. Due to integer math we have to launch more kernels in the case our computation array isn't % 16
	int const block = 16;
	parHost->blocks.x = (par->N + block-1)/block;
	parHost->blocks.y = (par->N + block-1)/block;
	parHost->threads.x = block;
	parHost->threads.y = block;
	
	//~ par->TotalWarp = parHost->blocks.x * parHost->blocks.y * 8;
	
	//For BatchFFT
	parHost->blocks1D = dim3(par->CN, 1);
	parHost->threads1D= dim3(MAXT, 1);
}

void ReadParamFromFile(ParD* par, ParHost* parHost){
	boost::property_tree::ptree pt;
	boost::property_tree::ini_parser::read_ini("Simulation/param.ini", pt);
	par->CN = pt.get<int>("main.CellNumber");
	par->N = pt.get<int>("main.N");
	par->L = pt.get<float>("main.L");
	par->dt = pt.get<float>("main.dt");
	par->alpha = pt.get<float>("main.alpha");
	par->beta = pt.get<float>("main.beta");
	
	par->k_a = pt.get<float>("main.k_a");
	par->k_b = pt.get<float>("main.k_b");
	par->k_c = pt.get<float>("main.k_c");
	par->KK_a = pt.get<float>("main.KK_a");
	par->rhotot = pt.get<float>("main.rhotot");
	par->DiffRD.x = pt.get<float>("main.D_rho");
	
	par->k_Ib = pt.get<float>("main.k_Ib");
	par->DiffRD.y = pt.get<float>("main.D_I");
	par->eta = pt.get<float>("main.eta");
	
	parHost->gamma = pt.get<float>("main.gamma");
	parHost->kappa = pt.get<float>("main.kappa");
	parHost->epsilon = pt.get<float>("main.epsilon");
	parHost->tao = pt.get<float>("main.tao");
	parHost->seed = pt.get<long>("main.seed");
	parHost->R = pt.get<float>("main.Radius");
	parHost->EndTime = pt.get<float>("main.EndTime");
	parHost->SaveTime = pt.get<float>("main.SaveTime");
	
	parHost->PlotStates = true;
}

void Scaling(ParD* par, ParHost* parHost){
	//phi
	par->alpha = par->alpha * par->dt/parHost->tao;
	std::cout << "par->alpha" << par->alpha << std::endl;
	par->beta  = par->beta * par->dt/parHost->tao;
	std::cout << "par->beta" << par->beta << std::endl;
	//~ par->grep  = par->grep * par->dt/(parHost->tao * parHost->epsilon);
	//~ par->sigma = par->sigma * par->dt * parHost->epsilon * parHost->epsilon/(parHost->tao * 12.0 * par->dx2);
	par->gamma = parHost->gamma * par->dt /(parHost->epsilon * parHost->epsilon)/parHost->tao;
	par->kappa = parHost->kappa * par->dt /(parHost->epsilon * parHost->epsilon)/parHost->tao;
	
	
	std::cout << "par->gamma" << par->gamma << std::endl;
	//rho
	par->k_b = par->k_b * par->dt;
	par->k_c = par->k_c * par->dt;
	par->KK_a = par->KK_a * par->KK_a;
	par->DiffRD.x = par->DiffRD.x * par->dt/(2 * par->dx2);
	
	//I
	par->k_Ib = par->k_Ib * par->dt;
	par->eta = par->eta * sqrt(par->dt)/par->dx;
	par->DiffRD.y = par->DiffRD.y * par->dt/(2 * par->dx2);

}

void SetUpPosM(dPointer *d_point, ParD* par, ParHost* parHost){
	float2 *ComPos	= new float2[par->N2];
	par->DeltaX = 2.0*M_PI/((float)par->N);
	
	for(size_t y=0; y<par->N; y++){
		for(size_t x=0; x<par->N; x++){
			size_t offset = x + par->N * y;
			ComPos[offset].x = cos(y*par->DeltaX)/par->DeltaX;
			ComPos[offset].y = sin(y*par->DeltaX)/par->DeltaX;
		}
	}
	
	CudaSafeCall( cudaMemcpy(d_point->ComPos, ComPos, par->N2*sizeof(float2), cudaMemcpyHostToDevice) );
	delete[] ComPos;
}

void SetUpSpecM(dPointer *d_point, ParD* par, ParHost* parHost){
	float qx[par->N], qy[par->N];
	
	for(int i=0; i<=par->N/2; i++){  //dk/2*i;
		qx[i]=parHost->dk*i;
		qy[i]=parHost->dk*i;
	}
	
	for(int i=1; i<par->N/2; i++){ //-dk/2*i;
		qx[par->N-i]=-parHost->dk*i;
		qy[par->N-i]=-parHost->dk*i;
	}
	
	float *SpectralDiffBend_h = new float[par->N2]();
	float4 *SpectralGradLap_h = new float4[par->N2]();
	
	float k2, k4, OperatorPF;
	for(int y=0; y<par->N; y++){
		for(int x=0; x<par->N; x++){
			int const offset = x+par->N*y;
			k2 = qx[x] * qx[x] + qy[y] * qy[y];
			k4 = (qx[x] * qx[x] + qy[y] * qy[y]) * (qx[x] * qx[x] + qy[y] * qy[y]);
			
			//~ OperatorPF = -par->dt * (parHost->gamma/parHost->tao * k2 + k4 * parHost->kappa/parHost->tao);
			OperatorPF = -par->dt * (parHost->gamma/parHost->tao * k2 + k4 * parHost->kappa/parHost->tao);
			
			SpectralDiffBend_h[offset] = exp(OperatorPF) /par->N2;
			
			SpectralGradLap_h[offset].x = -k2 /par->N2;
			
			//first Deriv
			float DivX = x;
			if(x == par->N/2){ 
				DivX = 0;
			}else if(x > par->N/2){
				DivX = x - par->N;
			}
			
			float DivY = y;
			if(y == par->N/2){ 
				DivY = 0;
			}else if(y > par->N/2){
				DivY = y - par->N;
			}
			
			DivX *= parHost->dk;
			DivY *= parHost->dk;
			
			SpectralGradLap_h[offset].y = 0;
			SpectralGradLap_h[offset].z = DivX /par->N2;
			SpectralGradLap_h[offset].w = DivY /par->N2;
			
		}
	}
	
	CudaSafeCall( cudaMemcpy(d_point->SpecMethDiffBend, SpectralDiffBend_h, par->N2*sizeof(float), cudaMemcpyHostToDevice) );
	CudaSafeCall( cudaMemcpy(d_point->SpectralGradLap, SpectralGradLap_h, par->N2*sizeof(float4), cudaMemcpyHostToDevice) );
	delete[] SpectralDiffBend_h;
	delete[] SpectralGradLap_h;
}

void WritePosition(ParD* par, ParHost* parHost, dPointer * d_point){
	float2 *Pos	= new float2[par->CN * par->NumSave];
	CudaSafeCall( cudaMemcpy(Pos , d_point->Pos , par->CN * par->NumSave * sizeof(float2), cudaMemcpyDeviceToHost) );
	
	std::string Myfile=parHost->path+"/Pos.dat";  			//add path to file
	std::ofstream outPos;
	//output for velocity and position
	outPos.open (Myfile.c_str(), std::ios::app);
	
	for(int t=0; t < par->NumSave; t++){
		outPos << parHost->SaveTime * t ;
		for(int CellIdx=0; CellIdx < par->CN; CellIdx++){
			outPos << "\t"<< Pos[CellIdx].x *par->dx << "\t" << Pos[CellIdx].y * par->dx;
		}
		outPos << std::endl;
	}
	outPos.close();
	
	delete[] Pos;
}