import torch

def noise(size, dim, cuda_idx = 0):
    n = torch.randn(size, dim)
    if torch.cuda.is_available(): return n.cuda(cuda_idx) 
    return n

def real_data_target(size, cuda_idx = 0):
    '''
    Tensor containing ones, with shape = size
    '''
    data = torch.ones(size, 1)
    if torch.cuda.is_available(): return data.cuda(cuda_idx)
    return data

def fake_data_target(size, cuda_idx = 0):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = torch.zeros(size, 1)
    if torch.cuda.is_available(): return data.cuda(cuda_idx)
    return data

def train_discriminator(discriminator, optimizer, loss, real_data, fake_data, cuda_idx=0):
    # Reset gradients
    optimizer.zero_grad()
    
    # Propagate real data
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, real_data_target(real_data.size(0), cuda_idx))
    error_real.backward()

    # Propagate fake data
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0), cuda_idx))
    error_fake.backward()
    
    # Take a step
    optimizer.step()
    
    # Return error
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(discriminator, optimizer, loss, fake_data, cuda_idx=0):
    # Reset gradients
    optimizer.zero_grad()

    # Propagate the fake data through the discriminator and backpropagate.
    # Note that since we want the generator to output something that gets
    # the discriminator to output a 1, we use the real data target here.
    prediction = discriminator(fake_data)
    error = loss(prediction, real_data_target(prediction.size(0), cuda_idx))
    error.backward()
    
    # Update weights with gradients
    optimizer.step()
    
    # Return error
    return error