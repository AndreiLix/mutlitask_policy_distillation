


small or no size mentioned: 
    policy_kwargs =dict(activation_fn=th.nn.Tanh, # Perhaps try ReLU
                         features_extractor_class=CustomCombinedExtractor,
                         features_extractor_kwargs=dict(state_embedding_mlp=[128, 128], task_embedding_mlp=[16, 32], activation=th.nn.Tanh),
                         net_arch=[dict(vf=[64, 64], pi=[64, 64])] )

big: 
    policy_kwargs = dict(activation_fn=th.nn.Tanh, # Perhaps try ReLU
                            features_extractor_class=CustomCombinedExtractor,
                            features_extractor_kwargr_kwargs=dict(state_embedding_mlp=[256, 128], task_embedding_mlp=[16, 32], activation=th.nn.Tanh),
                            net_arch=[dict(vf=[128], pi=[128])] )