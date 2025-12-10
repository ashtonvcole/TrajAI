import gns_tools as gt
import traj_ai as ta

def main():
    # Model parameters
    dt = 2 / 29.97 # Every other frame
    num_past = 5 # Consider past 5 frames, including present
    epsilon=1e-6
    dim_node = 8
    dim_edge = 8

    
    simulator = gt.GraphNeuralSimulator(
        graph_neural_network=gt.GraphNeuralNetwork(
            relater=NormalTangentialDistanceObjectiveStateRelater(
                num_past=num_past,
                epsilon=epsilon
            ),
            encoder_node=nn.Sequential(
                nn.ModuleList(
                    ta.NormalTangentialObjectiveStateTranscoder(
                        num_past,
                        epsilon
                    ),
                    gt.RectNN(
                        input_dim=dim_node,
                        output_dim=2,
                        width=16,
                        depth=2,
                        activation_type=nn.Tanh
                    )
                )
            ),
            encoder_edge=RectNN(
                input_dim=dim_node,
                output_dim=2,
                width=16,
                depth=2,
                activation_type=nn.Tanh
            ),
            processor=GraphNetwork(),
            decoder=gt.RectNN(
                input_dim=dim_node,
                output_dim=2,
                width=16,
                depth=2,
                activation_type=nn.Tanh
            )
        ),
        updater=ta.LocalAccelerationUpdater(
            dt=dt, 
            num_past=num_past
        )
    )

    # Training loop