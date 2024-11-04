// PetalAnimation.jsx
import { useEffect } from 'react';
import './PetalAnimation.css'; // Assuming you have specific styles for petals

const PetalAnimation = () => {
  useEffect(() => {
    const $petals = document.querySelectorAll('.petal');
    let colorIndex = 0;
    const colors = ['#FF6347', '#FF4500', '#FFD700', '#ADFF2F', '#00CED1']; // Example color array
    const cycleDuration = 7000; // Duration of one full cycle in milliseconds
    const startTime = Date.now();

    const updatePetalStyles = () => {
      const elapsed = Date.now() - startTime;
      const cycleProgress = Math.abs((elapsed % (2 * cycleDuration)) / cycleDuration - 1);
      colorIndex = Math.floor(cycleProgress * colors.length);

      $petals.forEach((b, ind) => {
        const i = ind / $petals.length;
        const x = Math.sin(i * Math.PI * 2) * 48;
        const y = -Math.cos(i * Math.PI * 2) * 48;
        const rotation = i * 360 + (cycleProgress * 360);

        const style = {
          transform: `translate(${x}px, ${y}px) rotate(${rotation}deg) scale(${0.5 + (cycleProgress * 0.5)})`,
          borderRadius: `${50 + (cycleProgress * 50)}% 0 ${50 + (cycleProgress * 50)}% 50%`
        };
        Object.assign(b.style, style);
        b.style.setProperty('--color', colors[colorIndex]);
      });
    };

    const animate = () => {
      updatePetalStyles();
      requestAnimationFrame(animate);
    };

    animate();

  }, []);

  return (
    <div style={{ position: 'relative', width: '100%', height: '200px' }}>
      {/* Petal Animation */}
      <div className="petal"></div>
      <div className="petal"></div>
      <div className="petal"></div>
      <div className="petal"></div>
      <div className="petal"></div>
      <div className="petal"></div>
      <div className="petal"></div>
      <div className="petal"></div>
    </div>
  );
};

export default PetalAnimation;
