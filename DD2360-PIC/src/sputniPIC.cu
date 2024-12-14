/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

#include <cuda_runtime.h>

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"


int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);
    
    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp= 0.0;
    
    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield field;
    field_allocate(&grd,&field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd,&ids[is],is);
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    
    // Allocate Particles
    particles *part = new particles[param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&part[is],is);
    }
    
    // Initialization
    initGEM(&param,&grd,&field,&field_aux,part,ids);
    
    
    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);
        
        
        
        // implicit mover
        // iMover = cpuSecond(); // start timer for mover
        // for (int is=0; is < param.ns; is++)
        //     mover_PC(&part[is],&field,&grd,&param);
        // eMover += (cpuSecond() - iMover); // stop timer for mover
        
        iMover = cpuSecond(); // start timer for mover
        FPfield *grd_XN_flat, *grd_YN_flat, *grd_ZN_flat, *field_Ex_flat, *field_Ey_flat, *field_Ez_flat, *field_Bxn_flat, *field_Byn_flat, *field_Bzn_flat;

        cudaMalloc(&grd_XN_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn);
        cudaMalloc(&grd_YN_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn);
        cudaMalloc(&grd_ZN_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn);
        cudaMalloc(&field_Ex_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn);
        cudaMalloc(&field_Ey_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn);
        cudaMalloc(&field_Ez_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn);
        cudaMalloc(&field_Bxn_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn);
        cudaMalloc(&field_Byn_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn);
        cudaMalloc(&field_Bzn_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn);

        cudaMemcpy(grd_XN_flat, grd.XN_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn, cudaMemcpyHostToDevice);
        cudaMemcpy(grd_YN_flat, grd.YN_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn, cudaMemcpyHostToDevice);
        cudaMemcpy(grd_ZN_flat, grd.ZN_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn, cudaMemcpyHostToDevice);
        cudaMemcpy(field_Ex_flat, field.Ex_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn, cudaMemcpyHostToDevice);
        cudaMemcpy(field_Ey_flat, field.Ey_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn, cudaMemcpyHostToDevice);
        cudaMemcpy(field_Ez_flat, field.Ez_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn, cudaMemcpyHostToDevice);
        cudaMemcpy(field_Bxn_flat, field.Bxn_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn, cudaMemcpyHostToDevice);
        cudaMemcpy(field_Byn_flat, field.Byn_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn, cudaMemcpyHostToDevice);
        cudaMemcpy(field_Bzn_flat, field.Bzn_flat, sizeof(FPfield) * grd.nxn * grd.nyn * grd.nzn, cudaMemcpyHostToDevice);

        
        for (int is=0; is < param.ns; is++) {
            std::cout << "***  MOVER with SUBCYCLYING "<< param.n_sub_cycles << " - species " << part[is].species_ID << " ***" << std::endl;
            FPpart *part_x, *part_y, *part_z, *part_u, *part_v, *part_w;

            cudaMalloc(&part_x, sizeof(FPpart) * part[is].npmax);
            cudaMalloc(&part_y, sizeof(FPpart) * part[is].npmax);
            cudaMalloc(&part_z, sizeof(FPpart) * part[is].npmax);
            cudaMalloc(&part_u, sizeof(FPpart) * part[is].npmax);
            cudaMalloc(&part_v, sizeof(FPpart) * part[is].npmax);
            cudaMalloc(&part_w, sizeof(FPpart) * part[is].npmax);

            cudaMemcpy(part_x, part[is].x, sizeof(FPpart) * part[is].npmax, cudaMemcpyHostToDevice);
            cudaMemcpy(part_y, part[is].y, sizeof(FPpart) * part[is].npmax, cudaMemcpyHostToDevice);
            cudaMemcpy(part_z, part[is].z, sizeof(FPpart) * part[is].npmax, cudaMemcpyHostToDevice);
            cudaMemcpy(part_u, part[is].u, sizeof(FPpart) * part[is].npmax, cudaMemcpyHostToDevice);
            cudaMemcpy(part_v, part[is].v, sizeof(FPpart) * part[is].npmax, cudaMemcpyHostToDevice);
            cudaMemcpy(part_w, part[is].w, sizeof(FPpart) * part[is].npmax, cudaMemcpyHostToDevice);

            int threadsPerBlock = 256;
            int blocksPerGrid = (part[is].nop + threadsPerBlock - 1) / threadsPerBlock;

            mover_PC_gpu<<<blocksPerGrid, threadsPerBlock>>>(part[is],field,grd,param, part_x, part_y, part_z, part_u, part_v, part_w, grd_XN_flat, grd_YN_flat,
                grd_ZN_flat, field_Ex_flat, field_Ey_flat, field_Ez_flat, field_Bxn_flat, field_Byn_flat, field_Bzn_flat);

            cudaMemcpy(part[is].x, part_x, sizeof(FPpart) * part[is].npmax, cudaMemcpyDeviceToHost);
            cudaMemcpy(part[is].y, part_y, sizeof(FPpart) * part[is].npmax, cudaMemcpyDeviceToHost);
            cudaMemcpy(part[is].z, part_z, sizeof(FPpart) * part[is].npmax, cudaMemcpyDeviceToHost);
            cudaMemcpy(part[is].u, part_u, sizeof(FPpart) * part[is].npmax, cudaMemcpyDeviceToHost);
            cudaMemcpy(part[is].v, part_v, sizeof(FPpart) * part[is].npmax, cudaMemcpyDeviceToHost);
            cudaMemcpy(part[is].w, part_w, sizeof(FPpart) * part[is].npmax, cudaMemcpyDeviceToHost);

            cudaFree(part_x);
            cudaFree(part_y);
            cudaFree(part_z);
            cudaFree(part_u);
            cudaFree(part_v);
            cudaFree(part_w);
        }


        cudaFree(grd_XN_flat);
        cudaFree(grd_YN_flat);
        cudaFree(grd_ZN_flat);
        cudaFree(field_Ex_flat);
        cudaFree(field_Ey_flat);
        cudaFree(field_Ez_flat);
        cudaFree(field_Bxn_flat);
        cudaFree(field_Byn_flat);
        cudaFree(field_Bzn_flat);

        
        eMover += (cpuSecond() - iMover); // stop timer for mover
        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++)
            interpP2G(&part[is],&ids[is],&grd);
        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&grd,&param);
        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&param);
        
        
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd,ids,&idn);
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
        
    
    }  // end of one PIC cycle
    
    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }
    
    
    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    // exit
    return 0;
}


