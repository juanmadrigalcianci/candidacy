%----------------------------------------
%preallocates the hyperparameters
%----------------------------------------
N=5*10^4; %number of samples
par=23; %parameters
nproc=28;
%----------------------------------------
%Saves true parameters
%----------------------------------------
X=zeros(1,par);
%xs = 51500 + Width of PML; zs = 57000 + Width of PML
X(1)=54000;
X(2)=59500;
%MATERIAL 1
X(3)=2571.3237764085998;
X(4)=6128.6721584572;
X(5)=3459.6956863028004;
%MATERIAL 2
X(6)=2426.41503611;
X(7)=6355.122874946;
X(8)=3799.8148401117;
%MATERIAL 3
X(9)=2520.1160128747;
X(10)=6799.444019500699;
X(11)=3823.0108244928997;
%MATERIAL 4
X(12)=2599.5359253555;
X(13)=6858.268976329799;
X(14)=3985.4579096294;
%MATERIAL 5
X(15)=2972.6688776596;
X(16)=7906.5743439161;
X(17)=4673.3570083854;
%MATERIAL 6
X(18)=3076.0623054293;
X(19)=8424.094384693199;
X(20)=4928.330648029099;
%MATERIAL 7
X(21)=3060.2828745172997;
X(22)=8434.0837880401;
X(23)=4999.1367830875;
p_true=X;
%----------------------------------------
% Creates priors according to Anamika
%----------------------------------------
vs_unpert  = [3529; 3705; 3882; 3911; 4705; 4794; 4823]'; % Unperturbed v_s
rho_unpert = [2500*ones(1,4) 2900*ones(1,3)];             % Dito rho

% Prescribe uncertainties in the material properties.

layers=7;
alpha = 0.05;   % Uncertainty in v_s (multiplicative)
aa = 1.64;       % Lower bound on v_p/v_s
bb = 1.78;       % Upper bound on v_p/v_s
rho_low = 0.9;  % Lower bound on multiplicative uncertainty factor in rho
rho_high = 1.1; % Upper dito
Rl=rho_unpert*rho_low;
Ru=rho_unpert*rho_high;
vs_L = vs_unpert.*(1-alpha);
vs_H = vs_unpert.*(1+alpha);
vp_L = vs_L*aa;
vp_H = vs_H*bb;
%defines  maximum b_int and minimum a_int values for the material properties
a_int=[Rl;vp_L;vs_L];a_int=a_int(:)';
b_int=[Ru;vp_H;vs_H];b_int=b_int(:)';

% adaptivity 0 no, 1 yes
adapt=1;

%----------------------------------------
%Obtains Data
%----------------------------------------

beta=[1,10,100,10&8];
Nt=length(beta);
nx=58;
nz=34;
dt=10;
nx_true=nx*4;
nz_true=nz*4;
obs=1700;
dt_true=10/4;
X=cell(Nt,1);
Ns=1;
%Checks if datafile exist
filename=('original_data.mat');
if exist(filename,'file')
    data=load(filename);
    d=data.d;
    %dm=data.dm;
else
    disp('Data file not found. Generating data...')
    %0 means no movie. 1 otherwise.
    d=specfem(p_true,nx_true,nz_true,dt_true,obs,nproc);
    save(filename,'d');
end
%Perturbs the data
[dn,dm]=size(d);
noise=1E-3;
dp=d;
d=d+noise*randn(dn,dm);
DT=(1/obs);
%---------------------------------------------------------------------
%
% solver stuff
%
%---------------------------------------------------------------------
a=[20000,20000,a_int];
b=[120000,75000,b_int];
aa=a(3:end);
bb=b(3:end);
prior=@(x) 1*(prod(unifpdf(x,a,b))>0);
lik=@(f) -0.5*DT*sum((f'-d).^2)/(noise^2);
post=@(x,f) lik(f)+log(prior(x));

%%---------------------------------------------------------------------
%
% PSM parameters
%
%%---------------------------------------------------------------------
M =1; %number of samples
y=zeros(M,par);
x=zeros(M,par);
R=zeros(M,length(d));
%samples first valueR

for i=1:M
x(i,:)=a+(b-a).*rand(1,par);
R(i,:)=specfem(x(i,:),nx,nz,dt,obs,nproc) ;
end
ratio=zeros(1,M);
for i=1:M
    px(1,i)=post(X{i}(1,:),R(i,:));
end
disp('Entered MH loop')
y=zeros(length(Nt),par);
R=zeros(length(Nt),length(d));
acpt=zeros(N,Nt);
S=cell(Nt,1);
%creates matrix of initial covariances
ss=[800 1000 5000 20000]   ;
for k=1:Nt
    S{k}=10*eye(par);
    S{k}(1,1)=ss(k);
    S{k}(2,2)=ss(k);
end


for j=1:N
    
    %obtains samples in parallel
    
    %generates a proposal for each temperature
    
for k=1:Nt-1
	y(k,:)=0*randn(1,par);
	while prior(y(k,:))==0
        y(k,:)=X{k}(j,:)+(S{k}*randn(par,1))';
	end
    end
    y(N,:)=a+(b-a).*rand(1,par);
    
    for i=1:Nt-1
        R(i,:)=specfem(y(i,:),nx,nz,dt,obs,nproc) ;
        py(j,i)=post(y(i,:),R(i,:));
    end
    py(j,Nt)=0;
    
    %accepts-rejects
    
    for i=1:Nt
        ratio(i)=min(0,py(j,i)/beta(i)-px(j,i)/beta(i));
        if log(rand)<ratio(i) && isnan((py(j,i)/beta(i))-px(j,i)/beta(i))==0 && prior(y(i,:))~=0
            X{i}(j+1,:)=y(i,:);
            px(j+1,i)=py(j,i);
            acpt(j,i)=acpt(j,i)+1;
        else
            X{i}(j+1,:)=X{i}(j,:);
            px(j+1,i)=px(j,i);
        end
    end
    
    disp([' ratio ',num2str(exp(ratio))])
    
    %swaps
    if mod(j,Ns)==0
        for k=1:Nt-1
            ak=px(j+1,k+1)/beta(k)+px(j+1,k)/beta(k+1)-px(j+1,k)/beta(k)-px(j+1,k+1)/beta(k+1);
            disp(['prob. of swapping between t' num2str(k),' and t',num2str(k+1),' is ',num2str(min(1,exp(ak)))])
            if ak>log(rand)
                disp(['swapped temps ' , num2str(k) ,' and ',num2str(k+1)])
                %changes posteriors
                p1=px(j+1,k);p2=px(j+1,k+1);
                px(j+1,k)=p2; px(j+1,k+1)=p1;
                %changes points
                p1=X{k}(j+1,:);   p2=X{k+1}(j+1,:);
                X{k}(j+1,:)=p2;   X{k+1}(j+1,:)=p1;
            end
        end
    end
    

        filename = 'PT_1.mat';
   	save(filename) 
    if mod(j,20)==0
        %tries adaptivity
        if adapt==1
            for k=1:1
                if rand<.5
                    try
                        S{k}=chol(eye(par)+cov(X{k}(j-19:j,:)));
                    catch
                        S{k}=eye(par);
                        S{k}(1,1)=ss(k);
                        S{k}(2,2)=ss(k);
                    end
                end
            end
        end
        
        
        
    end
    disp(['acceptance at iteration ',num2str(j)])
    disp(sum(acpt)/j)
    
    disp(['current ',num2str(X{1}(j+1,1:2))])
    
    
    
    
end


