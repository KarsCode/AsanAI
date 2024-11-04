import TextScrambleComponent from '../TextScramble/TextScramble'
import PetalAnimation from '../Petals/Petals'

const HomeText = () => {
    return (
        <div className='flex w-full items-center justify-between'>
            <div className='flex flex-col w-1/4 pt-16'>
                <div className='text-7xl font-semibold'>
                    AsanAI
                </div>
                <div className='text-lg'>
                    Transform your yoga journey with real-time, tailored guidance that helps you achieve flawless alignment in every pose. Experience a deeper connection between mind and body as you receive personalized corrections, empowering you to enhance your practice and reach new levels of balance and tranquility.
                </div>
            </div>
            <div className='flex flex-col gap-32'>
                <div>
                    <TextScrambleComponent />
                </div>
                <div>
                    <PetalAnimation />
                </div>


            </div>
            <div className='w-1/4 flex flex-col pt-16'>
                <div className='text-7xl font-semibold'>
                    Technology for Health.
                </div>
                <div>
                    Leveraging AI and machine learning, our innovative system estimates yoga poses, offering real-time corrections by comparing your posture to ideal models. Using advanced mathematical techniques, we ensure accurate guidance, helping you achieve perfect alignment and improve your practice effectively.
                </div>
            </div>

        </div>
    )
}

export default HomeText
