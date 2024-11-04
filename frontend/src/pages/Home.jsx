// pages/Home.jsx
import { useState } from 'react';
import HomeText from '../components/HomeText/HomeText';
import WebcamComponent from '../components/WebcamComponent/WebcamComponent';
 // Import the PoseCard component
import { Row, Col } from 'react-bootstrap';
import PoseCard from '../components/PoseCards/PoseCard';

const poses = [
    { name: 'Dog', description: 'A basic pose that stretches the entire body.' },
    { name: 'Tree', description: 'A balancing pose that improves focus and concentration.' },
    { name: 'Half-Standing Fold', description: "A powerful yoga pose."},
    { name: 'Mountain', description: "A powerful yoga pose."}
];

const Home = () => {
    const [selectedPose, setSelectedPose] = useState(null);

    const handlePoseSelect = (pose) => {
        // If the clicked pose is already selected, deselect it
        if (selectedPose?.name === pose.name) {
            setSelectedPose(null);
        } else {
            setSelectedPose(pose);
        }
    };

    return (
        <div className='flex flex-col p-24 items-center'>
            <div>
                <HomeText />
            </div>
            <div className='text-xl font-semibold pt-48 pb-4'>
                Select a Pose: 
            </div>
            <Row className="flex  gap-2">
                {poses.map((pose, index) => (

                    <Col key={index} className="mb-4" xs={12} sm={6} md={4}>
                        <PoseCard pose={pose} onSelect={handlePoseSelect}  selected={selectedPose?.name === pose.name} />
                    </Col>
                
                ))}
            </Row>
            <div className='pt-10'>
                {selectedPose && <WebcamComponent selectedPose = {selectedPose.name}/>}
            </div>

        </div>
    );
};

export default Home;
