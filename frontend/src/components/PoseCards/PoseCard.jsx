/* eslint-disable react/prop-types */
// components/PoseCard.jsx
import { Card } from 'react-bootstrap';

const PoseCard = ({ pose, onSelect, selected }) => {
    return (
        <Card
            style={{
                width: '18rem',
                cursor: 'pointer',
                backgroundColor: selected ? '#cce5ff' : 'white', // Light blue background for selected
                color: selected ? 'blue' : 'black' // Change text color for selected
            }}
            onClick={() => onSelect(pose)}
        >
            <Card.Body className='border b-4 p-2 border-black h-24'>
                <Card.Title className='font-extrabold'>{pose.name}</Card.Title>
                <Card.Text>
                    {pose.description}
                </Card.Text>
            </Card.Body>
        </Card>
    );
};

export default PoseCard;
